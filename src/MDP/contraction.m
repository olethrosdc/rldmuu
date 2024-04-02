## -*- Mode: octave -*-

## Here we are using a linear operator P
P =rand(2,2)
for k=1:2
  P(k,:) /= sum(P(k,:))
endfor

n = 100
X = rand(2, n)
gamma = 2
for t = 1:100
  plot(X(1,:), X(2,:), '*')
  axis([0,1,0,1])
  pause(0.1)
  X = gamma*P *X
endfor


## Contraction mapping for solving f(x) = x
##
## Consider $x^2 = x$

