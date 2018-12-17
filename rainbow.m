function [imageRGB] = rainbow(N,M)

    image = ones(N,M,3);
    for j=1:M
        image(:,j,1) = j/M;
    end

    for i=1:N
        image(i,:,2) = i/N;
    end

    imageRGB = hsv2rgb(image);

end

