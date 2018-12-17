function [image] = flagRGB(N,M,k)

    image = zeros(N,M,3);
    image(:,:,k) = 1;

end
