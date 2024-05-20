load res.mat
x=[];
a = find(alg_type(:,1)~=' ');   % non void data
% row1 = string(arrayfun(@(v)num2str(v),1:length(lambda_v(1,:)),'uni',0))
% row2 = string(arrayfun(@(v)num2str(v),lambda_v(1,:),'uni',0))
% labelArray = [row1; row2];
% tickLabels = strtrim(sprintf('%s\\newline%s\n', labelArray{:}));

delay_comb=[];
rhoce_comb=[];
for i = 1:length(lambda_v(1,:))
    x=[x i i];
    %x=[x lambda_v(1,i) lambda_v(1,i)]
    delay_comb = [delay_comb delay_old_v(:,i) delay_v(:,i)];
    rhoce_comb = [rhoce_comb rhoce_old_v(:,i) rhoce_v(:,i)];
end

plot(lambda_v(1,:))
ylabel('request rate (req/s)')
xlabel('placing round')
figure

plot(x, [delay_comb(a,:)*1000])
ylim([0 500])
ylabel('request delay (ms)')
xlabel('placing round')

% p = gca();
% p.XTick = 1:length(lambda_v(1,:));
% p.XTickLabel = tickLabels;
legend(alg_type(a,:));
grid

figure
plot(x,[rhoce_comb(a,:)])
ylim([0 1])
ylabel('cloud edge network load')
xlabel('placing round')
legend(alg_type(a,:));
grid

figure
plot([1:length(lambda_v)],[cost_v(a,:)])
ylabel('edge cost')
xlabel('placing round')
legend(alg_type(a,:));
grid

figure
plot([1:length(lambda_v)],[nmicros_v(a,:)])
ylabel('n. of edge microservices')
xlabel('placing round')
legend(alg_type(a,:));
grid







