load res.mat
x=[]
delay_comb=[];
rhoce_comb=[];
for i = 1:length(lambda_v(1,:))
    x=[x i i];
    delay_comb = [delay_comb delay_old_v(:,i) delay_v(:,i)];
    rhoce_comb = [rhoce_comb rhoce_old_v(:,i) rhoce_v(:,i)];
end
delay_comb * 1000; % millisec delay
plot(x, [delay_comb(1,:); delay_comb(2,:)])
ylim([0 0.3])
ylabel('request delay (ms)')
xlabel('placing round')
grid

figure
plot(x,rhoce_comb(1,:))
ylim([0 1])
ylabel('cloud edge network load')
xlabel('placing round')
grid






