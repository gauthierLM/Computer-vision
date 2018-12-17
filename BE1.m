%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% COMPUTER VISION AND IMAGE PROCESSING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Dimensions of standard images that will be created along this program
N = 256;
M = 512;


%% Question 1
% 'Load and display a grayscale image'

image = imread('BE CV OTSU/Images/SpainBeach.jpg');

figure;
im1 = imshow(image);

figure;
im2 = imshow(image(:,:,1));


%% Question 2
% 'Build the image of the grayscale illusion line'

max = 0.95;
min = 0.05;
image = greyscale(N,M,min,max);

figure;
imshow(image);


%% Question 3
% 'Build an matrix of black and white stripes with a variable width T'

T = N/40;
image = stripes(N,M,T,0);

figure;
subplot(2,2,1)
imshow(stripes(N,M,20,0));
subplot(2,2,2)
imshow(stripes(N,M,40,0));
subplot(2,2,3)
imshow(stripes(N,M,20,1));
subplot(2,2,4)
imshow(stripes(N,M,40,1));


%% Question 4
% 'Interpret and analyse RGB images' 

image1 = imread('BE CV OTSU/Images/Teinte.jpg');
im1R = image1;
im1R(:,:,2) = zeros(size(im1R(:,:,2)));
im1R(:,:,3) = zeros(size(im1R(:,:,3)));

im1G = image1;
im1G(:,:,1) = zeros(size(im1G(:,:,1)));
im1G(:,:,3) = zeros(size(im1G(:,:,3)));

im1B = image1;
im1B(:,:,1) = zeros(size(im1B(:,:,1)));
im1B(:,:,2) = zeros(size(im1B(:,:,2)));


image2 = imread('BE CV OTSU/Images/oeil.jpg');
im2R = image2;
im2R(:,:,2) = zeros(size(im2R(:,:,2)));
im2R(:,:,3) = zeros(size(im2R(:,:,3)));

im2G = image2;
im2G(:,:,1) = zeros(size(im2G(:,:,1)));
im2G(:,:,3) = zeros(size(im2G(:,:,3)));

im2B = image2;
im2B(:,:,1) = zeros(size(im2B(:,:,1)));
im2B(:,:,2) = zeros(size(im2B(:,:,2)));

image3 = imread('BE CV OTSU/Images/cargo.jpg');
im3R = image3;
im3R(:,:,2) = zeros(size(im3R(:,:,2)));
im3R(:,:,3) = zeros(size(im3R(:,:,3)));

im3G = image3;
im3G(:,:,1) = zeros(size(im3G(:,:,1)));
im3G(:,:,3) = zeros(size(im3G(:,:,3)));

im3B = image3;
im3B(:,:,1) = zeros(size(im3B(:,:,1)));
im3B(:,:,2) = zeros(size(im3B(:,:,2)));

figure;
subplot(3,3,1)
imshow(im1R);
subplot(3,3,2)
imshow(im1G);
subplot(3,3,3)
imshow(im1B);
subplot(3,3,4)
imshow(im2R);
subplot(3,3,5)
imshow(im2G);
subplot(3,3,6)
imshow(im2B);
subplot(3,3,7)
imshow(im3R);
subplot(3,3,8)
imshow(im3G);
subplot(3,3,9)
imshow(im3B);


%% Question 5
% 'Build and display the French flag'

flag = france(N,M);

figure;
imshow(flag);
            

%% Question 6
% 'Use HSV code to display the HSV color space'

image = imread('BE CV OTSU/Images/cargo.jpg');
cargoHSV = rgb2hsv(image);

figure;
subplot(2,2,1)
imshow(cargoHSV)
subplot(2,2,2)
imshow(cargoHSV(:,:,1))
subplot(2,2,3)
imshow(cargoHSV(:,:,2))
subplot(2,2,4)
imshow(cargoHSV(:,:,1))

imageRGB = rainbow(N,M);

figure;
imshow(imageRGB);


%% Question 7
% 'Determine the values of parameters alpha, beta and gamma to transform a RGB
% image into grayscale image'

imageR = flagRGB(N,M,1); 
imageG = flagRGB(N,M,2); 
imageB = flagRGB(N,M,3); 

imageGreyR = rgb2gray(imageR);
imageGreyG = rgb2gray(imageG);
imageGreyB = rgb2gray(imageB);

alpha = imageGreyR(1,1)/imageR(1,1,1);
beta = imageGreyG(1,1)/imageG(1,1,2);
gamma = imageGreyB(1,1)/imageB(1,1,3);

% alpha = 0.2989
% beta = 0.5870
% gamma = 0.1140


%% Question 9
% 'Work two images with histograms'

image1 = imread('BE CV OTSU/Images/imagexx.bmp');
image2 = imread('BE CV OTSU/Images/imagex.bmp');

hist1 = imhist(image1);
hist2 = imhist(image2);

figure;
imshow(image1);
figure;
imshow(image2);
figure;
plot(hist1)
figure;
plot(hist2)


%% Question 10
% 'Load ans display SpainBeach.png and isolate the beach'

image = imread('BE CV OTSU/Images/SpainBeach.jpg');

disk4 = fspecial('disk',4);
disk10 = fspecial('disk',10);
sobelH = fspecial('sobel');
sobelV = fspecial('sobel')';
gaussian5_2 = fspecial('gaussian',5,2);
kernel1 = [1,0,-1; 7,0,-7; 1,0,-1];
kernel2 = [1,7,1; 0,0,0; -1,-7,-1];

diskBlurred = imfilter(image,disk4);
sobelledH = imfilter(image,sobelH);
sobelledV = imfilter(image,sobelV);
gaussianed = imfilter(image,gaussian5_2);

imageHSV = rgb2hsv(image);
beach = imageHSV(:,:,1)<0.07;
beachFinal = imfilter(beach,disk10);

figure;
imshow(image);
figure;
imshow(beach);
figure;
imshow(beachFinal);


%% Question 12
% 'Isolate the main 5 stars of the image Etoile.png'

stars = imread('BE CV OTSU/Images/Etoiles.png');

starsHSV = rgb2hsv(stars);
starsVFilter = starsHSV(:,:,3) > 0.75;
starsVAverage = imfilter(starsVFilter,disk10);

imshow(stars);
figure;
imshow(starsHSV(:,:,3));
figure;
imshow(starsVFilter);
figure;
imshow(starsVAverage);


%% Question 13
% 'Get the FT and analyze the spectrum of images with stripes'

image = stripes(N,M,20,1);

G = fft(image(M/2,:));
G = fftshift(G);
H = fft(image(:,N/2));
H = fftshift(H);
I = fft2(image);
I = fftshift(I);

figure;
plot(abs(G));
figure;
plot(abs(H));
figure;
imagesc(abs(I))
figure;
imshow(ifftshift(ifft2(I)));


%% Question 14
% 'Blur the image with different kernels and interpret the spectrum'

image = stripes(N,M,20,1);
blurred4 = imfilter(image,disk4);
blurred10 = imfilter(image,disk10);

I1 = fftshift(fft2(image));
I2 = fftshift(fft2(blurred4));
I3 = fftshift(fft2(blurred10));

figure;
subplot(3,2,1)
imagesc(abs(I1));
subplot(3,2,3);
imagesc(abs(I2));
subplot(3,2,5);
imagesc(abs(I3));
subplot(3,2,2);
imshow(image);
subplot(3,2,4);
imshow(blurred4);
subplot(3,2,6);
imshow(blurred10);


%% Question 15
% 'Write a program that extracts the specific field in the image
% Champs.jpg'

image = imread('BE CV OTSU/Images/champs.png');
imageNormalized = double(image/255);

I = fftshift(fft2(imageNormalized(:,:,1)));
Iabs = abs(I);

n = size(I);
for i=1:n(1)
    for j=1:n(2)
        if (j-5.2/4*i+70)^2 >1000 ||  j<350 || j>370
            I(i,j) = 0;
        end
    end
end

finalImageGray = abs(ifft2(ifftshift(I)))*3;
finalImageGrayBlurred = imfilter(finalImageGray,disk4);
finalImage = finalImageGrayBlurred>0.2;
finalImageColor = double(image).*finalImage;

figure;
imshow(image);
figure;
subplot(1,2,1)
imagesc(log(Iabs));
colorbar
subplot(1,2,2)
imagesc(log(abs(I))); 
colorbar
figure;
imshow(finalImageGray);
figure;
imshow(finalImage);
figure;
imshow(uint8(finalImageColor));


%% Question 16
% 'What?s happen on the spectral domain when the blurring is implemented?'

image = stripes(N,M,20,1);
h7 = fspecial('average',7);
h5 = fspecial('average',5);
b = wgn(N,M,-100);

blurred7 = imfilter(image,h7);
blurred5 = imfilter(image,h5);

I = abs(fftshift(fft(image(N/2,:))));
I5 = abs(fftshift(fft(blurred5(N/2,:))));
I7 = abs(fftshift(fft(blurred7(N/2,:))));

figure;
subplot(1,3,1)
plot(I)
subplot(1,3,2) 
plot(I5);
subplot(1,3,3)
plot(I7)


%% Question 17
% 'Show that the function H(u) could be similar to a cardinal sinus by
% superposing the two functions. What conclusion could you give from this properties?
% Could you estimate T ?'

x = 1:1:512;

H = 1/7*(sin(pi*(x-256*ones(1,512))/512*7))./(sin(pi*(x-256*ones(1,512))/512));
Sc = sin(pi*(x-256*ones(1,512))/512*7)./(pi*(x-256*ones(1,512))/512*7);

figure;
plot(H);
hold on;
plot(Sc);

% Functions can be superposed enough to consider that Sc and H are similar.
% Thus, we can easily display when H is 0. (thanks to sinc properties)
% So, T can be estimated


%% Question 18
% 'Estimate T'

image = imread('toulouse.bmp');
x = 1:1:512;

% blurring with a 7*7 kernel
blurred7 = imfilter(image,h7);
I7 = fftshift(fft(image(M/2,:)));
Iabs7 = abs(I7);
y7 = blurred7;
J7 = fftshift(fft(y7(M/2,:)));
Jabs7 = abs(J7);
Habs7 = Jabs7./Iabs7;

% blurring with a 5*5 kernel
blurred5 = imfilter(image,h5);
I5 = fftshift(fft(image(N/2,:)));
Iabs5 = abs(I5);
y5 = blurred5;
J5 = fftshift(fft(y5(N/2,:)));
Jabs5 = abs(J5);
Habs5 = Jabs5./Iabs5;

I = fft2(image);
J = fft2(blurred7);
H = J./I;
Iabs = abs(I);
Jabs = abs(J);
Iest7 = uint8(ifft2(J));
Iest = uint8(ifft2(J./H));
Ireal = uint8(ifft2(I));

figure;
plot(log(Habs7));
A = abs(sinc(7/512*(x-256*ones(1,512))));
hold on;
plot(log(A))

figure;
plot(log(Habs5));
A = abs(sinc(5/512*(x-256*ones(1,512))));
hold on;
plot(log(A))

figure;
subplot(1,2,1)
imagesc(log(Iabs))
colorbar
subplot(1,2,2)
imagesc(log(Jabs))
colorbar

figure;
subplot(1,2,1)
imshow(image);
subplot(1,2,2)
imshow(y7);

figure;
subplot(1,4,1)
imshow(Iest7);
subplot(1,4,2)
imshow(Iest);
subplot(1,4,3)
imshow(Ireal);
subplot(1,4,4)
imshow(image);


%% Question 19
% 'Write a program to process inverse filtering method'

image = imread('toulouse.bmp');
I = fft2(image);

blurred7 = imfilter(image,h7);
Y = fft2(blurred7);

SeuilMax = 11 ; %beginning of the given program

hh = zeros(M);
centre = [1 1] + floor(M/2) ;
ext = (7-[1 1])/2;
ligs = centre(1) + (-ext(1):ext(1));
cols = centre(2) + (-ext(2):ext(2));

h = ones(7)/prod(7);
hh(ligs,cols) = h;
hh = ifftshift(hh);

H = fft2(hh);

ind = find(abs(H)<(1/SeuilMax));
H(ind) = (1/SeuilMax)*exp(j*angle(H(ind)));

G = ones(size(H))./H; % end of given program

Iest = G.*Y;
imageEst = uint8(ifft2(Iest));
imageBlurred = uint8(ifft2(Y));
imageReal = uint8(ifft2(I));

%% Question 20
% 'What?s happen with the image marcheur.jpg ?'





%% Question 21
% 'Restaure the blurred images and compare with the inverse filtering
% method'

image = imread('SpainBeach.jpg');
I = fft2(image);

blurred7 = imfilter(image,h7);
Y = fft2(blurred7);

RestW = deconvwnr(blurred7,h7,0);

figure;
subplot(1,3,1)
imshow(image);
subplot(1,3,2)
imshow(blurred7);
subplot(1,3,3)
imshow(RestW);


%%%%%%%%%%%---------END-----------%%%%%%%%%%%%%
