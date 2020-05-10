clear all
x = 2*rand([100000,2])-1;
y = x.*x;
g_1 = sum(x,2);
g_2 = -x(:,1).*x(:,2);
g_bar_1 = mean(g_1);
g_bar_2 = mean(g_2);

hold on;
plot([-1:0.01:1],[-1:0.01:1].*g_bar_1+g_bar_2);
plot([-1:0.01:1],[-1:0.01:1].^2);
legend({'$\bar{g}(x)$','$f(x)$'},'Interpreter','latex')

test_x = 2*rand(100000,1)-1;
g_bar = test_x*g_bar_1+g_bar_2;
bias = mean((g_bar - test_x.^2).^2);

g_x = test_x.*g_1+g_2;
var = mean((g_x - g_bar).^2);

E_out = mean((g_x - test_x.^2).^2);
