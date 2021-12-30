% train a feed-forward network with a single hidden layer to solve XOR
runs = 1;
success = 0;
rec_n = zeros(100, 1);
rec_success = zeros(100, 1);
rec_high_gamma = zeros(100, 2);
rec_high_direction = zeros(100, 2);
rec_no_iter_to_reach_threshold = zeros(100,1);
rec_no_gamma_calculations = zeros(100, 1);
rec_mean_step_size = zeros(100, 1);
while runs <= 100
    xi = [-1 -1 1 1; -1 1 -1 1];
    upsilon = [-1 1 1 -1];
    n1 = 2;
    makeNet(size(xi, 1), n1);
    % [theta, rec, reciterations_to_meet_threshold] = trainNet(xi, upsilon, n1);
    [theta, rec, rectheta, recp, recgamma, reciterations_to_meet_threshold] = myTrainNetCGS(xi, upsilon, n1);
    [val, idx] = max(recgamma(:, 1));
    rec_high_gamma(runs, :) = [val, idx];
    [val, idx] =max(rec(:, 2));
    rec_high_direction(runs, :) = [val, idx];
    rec_n(runs) = sum(rec(:, 4));
    rec_mean_step_size(runs) = mean((rec(:,2)).*(rec(:, 3)));
    rec_no_iter_to_reach_threshold(runs) = reciterations_to_meet_threshold;
    semilogy(rec(:, 1))
    y = zeros(size(upsilon));
    for k=1:size(xi, 2)
        y(k) = runNet([xi(:, k); theta; 0]');
    end
    if(isequal(upsilon, round(y)))
        rec_success(runs) = 1;
        success = success+1;
    else
        rec_success(runs) = -1;
    end
    rec_success(runs) = isequal(upsilon, round(y));
    [upsilon; y]
    runs = runs+1;
end
% mean(rec_n(:, 1))
% figure
% semilogy(rec_high_direction(:,1))
% figure
% semilogy(rec_high_direction(:,2))
% figure
% semilogy(rec_high_gamma(:,1))
figure
p(rec_mean_step_size(:,1))
