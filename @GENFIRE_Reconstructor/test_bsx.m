%% test the RBF kernel
% generate a random sequence of random points x and its function values f
n=50;
x = linspace(-pi/2,3*pi/2,n)' + 0.1*randn([n,1]); x = sort(x); x(1)=-pi/2;x(end)=1.5*pi;
f=sin(x) + 0.01*normrnd(0,1,[n,1]);
%% set up system A*phi = f and solve for kernel coefficient phi = A\f
% here I use RBF exp(-sigma*r), you can try different kernels
sigma = 0.25;
r = abs(bsxfun(@minus, x, x'));
A = exp(-sigma*r);
%A = sqrt(1 + .1*r);
phi = A\f;

%% approximate function at testing points s
s = linspace(-2*pi,3*pi,80)';
r2 = abs(bsxfun(@minus, s,x'));
B = exp(-sigma*r2);
fs = B*phi;
%% Inverse distance weighing (as used in GENFIRE)
f_inv = (1./r2)*f ./ (sum(1./r2,2));
%% plot result
figure;
hold on;
scatter(x,f);
plot(s,f_inv);
plot(s,fs);
legend('data','inverse distance','RBF');
hold off
% as you see that RBF performs better than inverse distance weighing
% also, I compute extrapolation points to see which one predicts better
% you can try RBF = sqrt(1 + sigma*r), this ones predicts very well
