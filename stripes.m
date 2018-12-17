function [image] = stripes(N,M,T,axis)
% im is a matrix of zeros of size N*M
    image = zeros(N,M);
    if axis == 1
        for i=1:M
            if mod(floor(i/T),2) == 0
                image(:,i) = 1;
            else
                image(:,i) = 0;
            end
        end
     else if axis == 0
        for j=1:N
            if mod(floor(j/T),2) == 0
                image(j,:) = 1;
            else 
                image(j,:) = 0;
            end
        end
    end
end

