function [image] = france(N,M)

    image = zeros(N,M,3);

    for i=1:floor(M/3)
        image(:,i,3) = 1;
    end

    for i=floor(2*M/3)+1:M
        image(:,i,1) = 1;
    end

    for i=floor(M/3)+1:floor(2*M/3)
        for j=1:3
            image(:,i,j) = 1;
        end
    end
end

