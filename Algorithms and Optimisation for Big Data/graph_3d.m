x=300;
k=linspace(300,310,81)
m = 10^8:10^8:(81)*10^8;

n=10000;
[K, M] = meshgrid(k, m);
Z = (1-exp(-K.*n*(1./M))).^K;

surf(K,M,Z)
title('Performance of bloom filter')
xlabel('k')
ylabel('m')
zlabel('False positive error rate')
colorbar