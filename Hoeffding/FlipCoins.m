clear all
data = zeros(100000,3);
for i = 1:100000
    flips = randi(2,10,1000) - 1;
    m = mean(flips);
    v_min = min(m);
    v_1 = m(1);
    v_rand = m(randi(1000));
    data(i,:) = [v_1, v_rand, v_min];
end

figure();
dat = hist(data,[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])/100000;
bar([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],dat);
legend('v_1',"v_{rand}","v_{min}");
xlabel("Fraction of heads in 10 flips");
ylabel("Probability");

d = zeros(101,4);
for x = 0:100
    d(x+1,1)=sum(abs(data(:,1)-0.5)>(x/100))/100000;
    d(x+1,2)=sum(abs(data(:,2)-0.5)>(x/100))/100000;
    d(x+1,3)=sum(abs(data(:,3)-0.5)>(x/100))/100000;
    d(x+1,4)=2*exp((-2)*((x/100)^2)*10);
end

figure();
plot([0:0.01:1],d);
legend('P[|v_1-\mu_1|>\epsilon]','P[|v_{rand}-\mu_{rand}|>\epsilon]',"P[|v_{min}-\mu_{min}|>\epsilon]","2e^{-2\epsilon^2N}");
xlabel("\epsilon");
ylabel("Probability");