function [image] = greyscale(N,M,min,max)

    image = zeros(N,M);
    bdMin = 0.4*N;
    bdMax = 0.6*N;

    for i = 1:N
        for j=1:M
            if i<bdMin || i>bdMax
                image(i,j) = min + j*(max-min)/M;
            else
                image(i,j) = (max-min)/2;
            end
        end
    end

end

