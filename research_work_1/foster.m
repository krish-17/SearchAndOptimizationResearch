function [f, d, H, isM] = foster(x, pk, pl)
n = size(x, 1);
A = diag(logspace(0, 2, n));
f = x.'*A*x;
d = 2*A*x;
H = 2*A;
if exist('pk','var')
    isM = pk*A*pl.';
end
end

