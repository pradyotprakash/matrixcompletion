clc; clear all; close all;
% Creating random matrix data
n=500;
d=10;
P=randn(n,d)
Q=randn(n,d);
X=P*Q'; % 500 x 500 matrix

% samples
sampfrac=.5;
m=floor(sampfrac*n^2);
%observed 
omega = randsample(n^2, m);% returns m x1 vector - uniform dist over 1 to n^2

% observations
Y = zeros(n);

Y(omega)=X(omega);

Y0 = Y; %Saving incomplete matrix

% Using ISVT - making rank 10
T = ceil(50/sampfrac) % Running the loop T times
error = zeros(1,T);
for i=1:T
    [u, s, v] = svd(Y);
    u=u(:,1:d);    
    s=s(1:d,1:d);
    v=v(:,1:d);
    Y=u*s*v';
    Y(omega)=X(omega);
    
    error(i) = norm(Y-X,'fro');
    if(error(i) < 0.01)
        error(i)
        fprintf('Converged at %d \n',i);
        break;
    end
end
