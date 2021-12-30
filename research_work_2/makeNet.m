function makeNet(n0, n1)
% Generate a feed-forward network with a single hidden layer

  x0 = sym('x0', [n0, 1]);          % inputs
  x1 = sym('x1', [n1, 1]);          % hidden layer
  y = sym('y');                     % output

  upsilon = sym('upsilon');         % desired output
  n = n0*n1+2*n1+1;                 % number of parameters. Why there is 2 for the hidden layer?
  theta = sym('theta', [n, 1]);     % weights and biases
  s = sym('s');

  % compute the outputs of the hidden layer
  for k=1:n1
    s = theta(n0*(k-1)+1:n0*k).'*x0-theta(n0*n1+n1+k);
    x1(k) = tanh(s);
  end

  % compute the output of the output layer
  s = theta(n0*n1+1:(n0+1)*n1).'*x1-theta(end);
  y = tanh(s);

  % compute the loss
  f = -log((1+sign(upsilon)*y)/2);

  % symbolically compute the gradient
  grad = sym('grad', [n, 1]);
  for k=1:n
    grad(k) = diff(f, theta(k));
  end

  % generate a function file
  xIn = num2cell(x0);
  thetaIn = num2cell(theta);
  matlabFunction(y, f, grad, ...
    'File', 'runNet', ...
    'Vars', {[xIn.', thetaIn.', upsilon]});

end