function [theta, rec, rectheta, recp, recgamma, reciterations_to_meet_threshold] = myTrainNetCGS(xi, upsilon, n1)
    [n0, m] = size(xi);
    n = n0*n1+2*n1+1;
    % how the values are initialized
    % theta comprises of the weights and bias
    theta = [randn(n-n1-1, 1); zeros(n1+1, 1)];
    % Theta has always last three rows as 0.
    f = zeros(1, m);
    y = zeros(1, m);
    grad = zeros(n, m);
    rec = zeros(1000, 5);
    rectheta = zeros(1000, 9);
    recp = zeros(1000, 9);
    recgamma = zeros(1000, 1);
    reciterations_to_meet_threshold = 0;
    restart = 2'm;
    for k=1:m
       [y(k), f(k), grad(:, k)] = runNet([xi(:, k); theta; upsilon(k)]');
    end
    p = mean(grad, 2);
    p0 = -p;
    for t=1:200
        [gamma, n, c] = ags(theta, p0, 0, 1, 1.0e-08, upsilon, m, xi);
        if (isinf(gamma) || isnan(gamma))
            break;
        end
        newtheta = theta + gamma * p0; % determining the minimum value in the direction computed
        % implementation of the fletcher reeves begin
        if(mod(t, restart) == 0)
            for k=1:m
                [~, f(k), grad(:, k)] = runNet([xi(:, k); newtheta; upsilon(k)]');
            end
            pn = -(mean(grad, 2));
        else
            for k=1:m
                [~, ~, grad(:, k)] = runNet([xi(:, k); newtheta; upsilon(k)]');
            end
            dk = mean(grad, 2); % gradient of the kth term
            numerator = dk.'*dk; % numerator of fletcher reeves
            for k=1:m
                [~, f(k), grad(:, k)] = runNet([xi(:, k); theta; upsilon(k)]');
            end
            dkm1 = mean(grad, 2); % gradient of k-1 term
            denominator = dkm1.'*dkm1; % denominator of fletcher reeves
            pn = -dk + ((numerator/denominator)*(p0)); % New direction of the minimum
        end
        rectheta(t, :) = theta;
        recp(t, :) = p0;
        recgamma(t, :) = gamma;
        rec(t, :) = [mean(f), norm(p0), gamma, n, c];
        p0 = pn;
        theta = newtheta;
        if(mean(f) <= .1)
            break
        end
        reciterations_to_meet_threshold = reciterations_to_meet_threshold + 1;
        % updating current direction as new direction and moving forward
        % for the next iteration
    end
end