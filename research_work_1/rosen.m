function [y, g, fOpt] = rosen(x)
    z = 1*x;
    d = sum(100*(z(2:end)-z(1:end-1).^2).^2 + (z(1:end-1)-1).^2);
    y = d^1;
    g1 = -400*z(1:end-1).*(z(2:end)-z(1:end-1).^2) - 2*(1-z(1:end-1));
    g2 = 200*(z(2:end)-z(1:end-1).^2);
    g = [g1; 0] + [0; g2];
    g = 1*d^(1-1)*1.'*g;
    fOpt = 0;
end