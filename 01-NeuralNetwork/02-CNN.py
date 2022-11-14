'''
we have 1000*1000 pixel image as an input and each pixel is an input so. and imagine we have a hidden layer with 10^6 units --> how many weight do you think we need?

    x1                  z1|a1
    x2                  z2|a2
    .                   .
    .                   .
    .                   .
    x999999             z999999|a999999
    x1000000            z1000000|a1000000


we need 10^6 * 10^6 weights !!!! ---> so feedforward NN is useless in this context (for images I mean)

so we use two things to overcome this difficulty ---> use a filter of shape 11*11 for example --> which is like a square patch striding through the image
so now we have 11*11=121 weights parameter that operate on the entire image

another thing is feedforward NN is very sensitive about where the mushroom or cat or etc is. but in CNN we use a pooling procedure which give us translational invariance 
so it will detect the mushroom for test data regardless of location



convolution --> continuous case: intuitively you are flipping g over y axis to get g(-tau)  and shift it  over f by replacing tau by (tau-t) and you get g(t-tau) which means shifting g(-tau) to the right by the amount of t. remember to view t as a constant
then do the product and then find the are under the new curve.
some good explanation here ---> https://www.youtube.com/watch?v=zQ7Khy-MifQ&ab_channel=BarryVanVeen

            (f*g)(t) = integral OF  f(tau)g(t-tau)d_tau and tau=-inf to +inf

imagine two squares and you first flip one and sweep it from -inf to +inf and at some point it goes from the other square ---> as the overlap increases the conv increases and as it goes out
of the square conv decreases so the convolution will be a triangle


discrete case:
(f*g)[n] =sum of f[m]g[n-m] and m is from -inf to +inf

cross-correlation as equivalent of convolution in CNN:
There is another thing called cross-correlation which is f.g(t) = sum of f(tau) g(t+tau) and tau=-inf to +inf --> its the same as conv but there is no flipping of g happening here
and for the 2D case we have

f.g(x,y) = sum sum f(tau1,tau2) g(tau1+x , tau2+y) tau1 and 2 is from -inf to +inf ----> so you actually fix f and make the g to shift

Example : f[k] = [1,2,3] , g[k] = [2,1] and suppose  k starts from 0 what is h[k] = f[k] * g[k] = sum f[tau]g[k-tau] and tau=-inf to inf and put 0 where f and g are not defined --> zero padding

for example for h[0]:
flipped g'...0 1 2 0 ...     ----> we call the flipped g' as filter or kernel , for the input signal or image f
        f   ...0 1 2 3 0 ...
        h        2


for h[1]

flipped g'  ...0 1 2 0 ...
        f   ...0 1 2 3 0 ...
        h        1+4 = 5




Example --> f=[1,3,-1,1,-3] g'=[1,0,-1] --> g' is the flipped g

f*g ---> [2,2,2]  --> without zero padding

for zero padding we use f = [0,1,3,-1,1,-3,0] --> f is the input
f*g = [-3,2,2,2,1] --> now the output dim is the same as the input dim




Example 2D discrete: image f and a filter g'

f = [1 2 1
     2 1 1
     1 1 1]

g' = [1 0.5
      0.5 1]


We align the filter with the top left corner of the image, and take the element wise multiplication of the filter and
the 2 by 2 square in the top left corner. We then shift the filter along the top row, doing the same thing. We then
apply the same procedure to the next row. If we went another row down, the bottom row of the filter would not
have any numbers to be multiplied with. Thus, we stop

      conv = [4 4
              4 3]
so the sum is 15

g' is the filter and its the model learned in CNN and its not given. so it doesn't matter if we flip the filter or not so we use cross-correlation instead of convolution
so in cnn f is the signal ---> if its an image its a 2D signal and g is the filter



pool:
A pooling layer finds the maximum value over a given area. The max value can be seen as a "signal" representing
whether or not the feature exists. For example, a high max value could indicate that the feature did appear in the
image(regardless of where it appears in the image).



output if CNN:
Pool(ReLU(Conv(Image))) where ReLU(x) = max(0,x) and assume a stride for the conv and pool layer --> for example stride = 1

Example : image I =[1 0 2
                    3 1 0
                    0 0 4]
 filter weight F = [1 0    
                    0 1]
   output of cnn = Pool(ReLU(Conv(I)))

   conv(I) = [2 0
              3 5]

    ReLU(Conv(I)) = ReLU([2 0; 3 5]) ---> [2 0 ; 3 5]
    Pool([2 0 ; 3 5]) = 5

Each filter represents a distinct set of weights, which corresponds to searching for a particular feature in the
image. If you have a large number of features, you want many filters.
'''