import praw
import re
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import itertools

import sys, os
from datetime import datetime,timedelta

from temporal import TGraph

from psaw import PushshiftAPI

reddit = praw.Reddit('bot', user_agent='test script by u/research-oli')
papi = PushshiftAPI() # pass in reddit, but this may have a bug


sub_pattern = re.compile(r'r/(\w*)')
def extract_subs(text):
    return set(sub.lower() for sub in sub_pattern.findall(text)) - {''}


def from_popular_subs(N = 100):
    G = nx.DiGraph()

    for s in reddit.subreddits.popular(limit=N):
        G.add_node(s.display_name, sub=s)

    return G

############### FROM SUB DESCRIPTION #################
def buildEdges(G, sub2subs, mode = 'clip'):
    todo = G    
    while todo:
        print("\n"+ '*'*25+ 'WORK' + '*'*25+"\n|V| = "+str(len(G))+";  \t|E| = "+str(len(G.edges)) \
            + ";   \t |TODO| = "+str(len(todo)) )
        work = list(todo)
        todo = []
        for sn in work:
            s = reddit.subreddit(sn)
            for tn, edgedata in sub2subs(s).items():
                if tn not in G and mode == 'clip':
                    continue
                
                # G.add_node(tn)
                G.add_edge(sn, tn, **edgedata)
                    
                if mode == 'iterate':
                    todo.append(tn)
    return G


########### FROM DESCRIPTION ###################
def sub2descrsubs(sub):
    try :
        descr = sub.description
    except:
        return {}
    
    # print(sub,end=';')
    return { sn : dict(weight=1) for sn in extract_subs(descr) }
    

########## FROM REDIRECT TOP LEVEL RESPONSES TO TOP POSTS ###########
def sub2redirectsubs(sub, n_posts=30, n_comments=None):
    subs = {}
    for post in sub.top('month', limit=n_posts):
        # print(post.title[:80])
        for c in itertools.islice(post.comments, n_comments):
            if isinstance(c, praw.models.Comment):
                # print('\t\t'+c.body[:40].replace('\n','\\n'))
                for t in extract_subs(c.body):
                    subs[t] = dict(weight = c.score)
            
    return subs


def make_postshift_graph(
        N = 300, 
        start_ts = datetime(2017,1,1),
        end_ts = datetime(2019,1,1),
        dt = timedelta(days=1),
        min_comments = 10,
        comment_support = 40,
        graph_cache = None
):
    ############# CONVERT TIMES #############
    y_start = y_end = ''
    if type(start_ts) == tuple:
        start_ts = dt.datetime(*start_ts)
    if type(end_ts) == tuple:
        end_ts = dt.datetime(*end_ts)    
    if type(start_ts) == datetime :
        y_start = str(start_ts.year)[2:]
        start_ts = int(start_ts.timestamp())
    if type(end_ts) == datetime :
        y_end = str(end_ts.year)[2:]
        end_ts = int(end_ts.timestamp())
    if type(dt) == timedelta:
        if dt.seconds % 43200 == 0:
            if dt.days % 7 == 0:
                dt_str = str(dt.days//7) + 'W'
            else:
                dt_str = str(dt.days) + 'D'
        dt =  int(dt.total_seconds())
    else:
        dt_str = str(dt) + 'S'
     # force day, because we cangraphs't query at every aggregation step.
    #########################################
    
    # T = (end_ts - start_ts) // dt
    
    # agg_ncomments = next(papi.search_submissionstr(datetime(2017,1,1).year)[2:] s(before=end_ts, after=start_ts, aggs='subreddit',\
    #    num_comments='>'+str(min_comments), agg_size=N, limit=0))['subreddit']      
    
    # sub_names = [ d['key'] for d in agg_ncomments ] # cannoncial ording on subs, forwards
    # sub_idx = { n : i for i,n in enumerate(sub_names) } # backwards lookup on subs. Update as we add more.
    # U = np.zeros((T,N))
    
    if graph_cache is None:
        graph_cache = []
    tracked_subs = set()
    
    terminal_cols = os.get_terminal_size(0)[0]
    
    for d in next(papi.search_submissions(before=end_ts, after=start_ts, aggs='subreddit',\
            num_comments='>'+str(min_comments), agg_size=(N), size=0))['subreddit']:
        tracked_subs.add(d['key'])
    
    
    # option 1 : loop through subs, aggregate based on time.
    # for sub,i in sub_idx.items():

    # option 2 : loop though times, aggregate based on sub.
    for t in range(start_ts, end_ts,dt): 
        print(datetime.fromtimestamp(t), '   fetching activity...', end='', flush=True)
        # find all submissions in time window with more than 10 comments; aggregate by subredit.
        comments_agg = next(papi.search_submissions(before=t+dt, after=t, aggs='subreddit',
             num_comments='>'+str(min_comments), agg_size=(N*2), limit=0,
             subreddit=','.join(tracked_subs)
        ))['subreddit']
        
        G = nx.DiGraph()
        
        for d in comments_agg:
            sub = d['key']
            G.add_node(sub, activity=d['doc_count'])
        
        for i,v in enumerate(tracked_subs):
            sys.stdout.write('\r'+ str(datetime.fromtimestamp(t))+ '   |' + '='*(i)+ '>' + \
                '-'*(len(tracked_subs)-i-1)+"|"+' '*17+"\n\t")
            
            
            if comment_support is None:
                batch_request = papi.search_comments(q='r/'+v, before=t+dt, after=t, 
                    aggs='subreddit', subreddit=','.join(tracked_subs), agg_size=(N*2))
                link_counts = { (d['key'],v) : d['doc_count'] for d in next(batch_request)['subreddit'] }
                nx.set_edge_attributes(G, 'nlinks', link_counts)

                ####### THIS IS ERROR CHECKING CODE #####
                ##### I have already verified that the reason for misses is non-existence.
                # found = { l[0] for l in link_counts }
                # missed = tracked_subs - found
                # if missed:
                #     picked_up = {}
                #     for m in missed:
                #         more_search = papi.search_comments(q='r/'+v, before=t+dt, after=t, subreddit=m, 
                #         aggs='subreddit')
                #         try:
                #             picked_up[m] = next(more_search)['subreddit'][0]['doc_count']
                #         except:
                #             pass
                #     print("\nFound %d in batch: "%len(found), found, "\nMissed %d in batch: "%len(missed), missed, "picked up : ", picked_up, '\n')
            else:
                for u in tracked_subs:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                    
                    srch_results = papi.search_comments(q='r/'+v, before=t+dt, after=t, subreddit=u, 
                        aggs='subreddit', sort_type='score', limit=comment_support, filter='id')
                
                    try:
                        n_links = next(srch_results)['subreddit'][0]['doc_count']
                        top_links = [c.id for c in srch_results]
                            
                        G.add_edge(u,v, nlinks = n_links, toplinkingcomments = top_links)
                    except IndexError:
                        pass
            
            print('\r\t' + ' '*len(tracked_subs), end='', flush=True)
            if i + 1 == len(tracked_subs):
                sys.stdout.write('\r' + ' '*terminal_cols + '\r')
            else:
                sys.stdout.write("\033[F"*(1 + (len(tracked_subs)+25)//terminal_cols))
            
        # U = { d['key'] : d['doc_count'] for d in agg_ncomments }
        
        graph_cache.append(G)
        # Gs, times, filename, node_attrs, edge_attrs,
        return TGraph( graph_cache, list(range(start_ts, end_ts,dt)),
            "G%s_%d" % (dt_str,len(tracked_subs)) + '(%s-%s)'%(y_start,y_end) if y_start and y_end else '',
            ["activity"], ["nlinks", "toplinkingcomments"])


        
                                      


# def test_print(N):
#     print("asdf\nfdsa\nasdf");
#     import time
#     time.sleep(10)
#     sys.stdout.write("\033[F"*2)
#     print("oiiiiiiiiiii")
