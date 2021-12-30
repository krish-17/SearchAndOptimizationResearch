function [recn, recfx, recx, recp, recg] = conjugategradsearch(testFunc, x)
%set the dimenstion
dim = 100;
iterations = 100;
% It initializes the x0
 if nargin<2
    x = randn(dim, 1);
 end
% It stores the result at the end of each iteration
recn = zeros(iterations, 1);
k = 1;
%record function values at each iteration
recfx = zeros(iterations, 1);
recg = zeros(iterations, 1);
recx = zeros(iterations, dim);
recp = zeros(iterations, dim);
% applying the test function, in this case its ostermeier ellipsoid
[fx, p0] = testFunc(x);
recx(k, :) = x;
recfx(k, :) = fx;
% The negative of the gradient gives the p0. First step of Fletcher and
% Reeves.
p0 = -p0;
recp(k, :) = p0;
% iterating till the last dimension, in this case it is 10.
while true
    if k > iterations
        break;
     end
     % The function here is f(xK+1) = xk + gamma * p0;
     gf = @(g) testFunc(x + g*p0);
     % The function is passed to golden section search. a is set to 0 and c
     % is set to 270 initially.
     [gamma, n, ~] = ags(gf, 0, 1, 1.0e-40);
     recg(k,:) = gamma;
     recn(k, :) = n;
     % the xmin calculated is the gamma we are looking for. updating the
     % xk+1 by applying the above function in the direction p0.
     newx = x + (gamma*p0);
     % Implementation of Fletcher-Reeves starts here
     % the pk term in the equation
     [~, dk] = testFunc(newx); % delta f(xk)
     % second term numerator
     numerator = dk.'*dk;
     % second components denominator
     [~, dkm1] = testFunc(x); % delta f(xk-1)
     denominator = dkm1.'*dkm1;
     % The new direction is updated
     % the gradient returned by the test function is positive. We need the
     % negative of it
     pn = -dk + ((numerator/denominator)*(p0)); 
     % next iteration
     k = k + 1;
     % the values of this iteration is recorded
     recx(k, :) = newx;
     recp(k, :) = pn;
     recfx(k, :) = testFunc(newx);
     % reset the p and x for the next iteration of the loop
     p0 = pn;
     x = newx;
end
end