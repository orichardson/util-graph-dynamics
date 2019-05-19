###  Plan: 
from reddit_graph_builders import *

sub2redirectsubs(reddit.subreddit("murderedbywords"),20,None)
extract_subs(reddit.subreddit("funny").description)

G = from_popular_subs(100)
buildEdges(G, sub2descrsubs, "clip")

G.edges
len(G.edges)
nx.draw(G)

degree = G.degree()
G2 = G.subgraph([n for n in G if degree[n] > 0])
nx.draw(G2)

# wids = ask_sub.widgets
# rulewidget = wids.sidebar[0]
# get_subs(rulewidget.data)
# 
# praw.models.
# 
# for widget in wids.items:
#     print(wids[widget])
#     if isinstance(widget, praw.models.CommunityList):
#         print(widget)

reddit.subreddits.recommended([reddit.subreddit('askreddit'),reddit.subreddit('lifeprotips')])
top_6_subs = list(reddit.subreddits.popular(limit=6))

reddit.subreddits.recommended(top_6_subs)    

widgets = reddit.subreddit('askouija').widgets
for widget in widgets.sidebar:
    if isinstance(widget, praw.models.CommunityList):
        print(widget)
    

#####################

from convokit import Corpus, download

smalsubs = Corpus(filename=download('reddit-corpus-small')) # will not download twice if it already exists

ut_ids = smalsubs.get_utterance_ids()

len(ut_ids)

uid = ut_ids[0]

c_ids = smalsubs.get_conversation_ids()
cid = c_ids[0]

convo = smalsubs.get_conversation(cid)
convo.get_utterance_ids()
convo.get_utterance(uid)

top_level = [uid for uid in ut_ids if sub_corn.get]
