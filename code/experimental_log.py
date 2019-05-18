plot_errors(Er_static, times=G1W_100.times, max_bars=10, cmap=['gray']*2, show=False, msize=2)
plot_errors(Er_choice, times=G1W_100.times, max_bars=10, cmap='winter',reuse=True, msize=4)
Er_choice = calc_err(Qs, lambda Q, U : trans_alt(Q,U), Us, 6)
Er_static = calc_err(Qs, lambda Q: Q, nsamples=6)
pow100_Er = calc_err(Qs, lambda Q: linalg.matrix_power(Q,100))
npowerEr = [[[ linalg.norm( linalg.matrix_power(Q2, i) - Q)  for Q in Qs] for i in range(10)] for Q2 in Qs]
