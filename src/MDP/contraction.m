## -*- Mode: octave -*-

## Here we are using a linear operator P

# Random P
P =rand(2,2)
P = 0.1 + 0.9*eye(2)
for k=1:2
  P(k,:) /= sum(P(k,:))
endfor

T=100
n = 100
X = randn(2, n)
gamma = 0.9
for t = 1:T
  plot(X(1,:), X(2,:), '*')
  axis([-1,1,-1,1])
  pause(0.1)
  X = gamma*P *X;
endfor


## Contraction mapping for solving f(x) = sqrt(x)
##
## Consider $x^2 = x$
##
T=100
x_0 = 3
x = x_0
X = zeros(1, T)
c=1
for t = 1:T
  x = (c * x + x_0/x) / (1 + c)
  X(t) = x
endfor
plot(X)
pause
