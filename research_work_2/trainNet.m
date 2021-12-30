function [theta, rec, reciterations_to_meet_threshold] = trainNet(xi, upsilon, n1)

    [n0, m] = size(xi);
    n = n0*n1+2*n1+1;
    % how the values are initialized
    % theta comprises of the weights and bias
    theta = [randn(n-n1-1, 1); zeros(n1+1, 1)];
    % Theta has always last three rows as 0.
    f = zeros(1, m);
    y = zeros(1, m);
    grad = zeros(n, m);

    eta = 2.0;
    rec = zeros(1000, 3);
    reciterations_to_meet_threshold = 0;
    for t=1:1000
        for k=1:m
            [y(k), f(k), grad(:, k)] = runNet([xi(:, k); theta; upsilon(k)]');
        end
        % The dimension of the gradient vector is 9*4 for the inputs. How?
        % the plain gradient function starts here. Replacing here with the 
        % cojugate gradient method.
        % size of f is 1 * 4. This is the size of the boolean string.
        rec(t, :) = [mean(f), norm((mean(grad, 2))), eta];
        % xkp1 = xk - (gamma) * gradient;
        % 2 maps to the gamma here. why mean is used for gradient?
        % dimenstion of theta is 9*1.
        if(mean(f) <= .001)
            break
        end
        reciterations_to_meet_threshold = reciterations_to_meet_threshold + 1;
        theta = theta-eta*mean(grad, 2);
    end

end