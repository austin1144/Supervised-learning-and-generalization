clear
clc
close all

%%%%%%%%%%%
%algorlm.m
% A script comparing performance of 'trainlm' and 'traingd'
% traingd - batch gradient descent 
% trainlm - Levenberg - Marquardt
%%%%%%%%%%%

%generation of examples and targets
%x=0:0.01:3*pi; y=sin(x.^2)+rand(1);
% x=0:0.05:3*pi; y=sin(abs(x))+0.2*rands(1,189);
x=0:0.05:3*pi; y=sin(x);
p=con2seq(x); t=con2seq(y); % convert the data to a useful format
neuron=5;
error =0;
training_time=1000;
%creation of networks

net1=feedforwardnet(neuron,'trainlm'); 
net2=feedforwardnet(neuron,'trainlm'); 
net3=feedforwardnet(neuron,'trainlm'); 
net4=feedforwardnet(neuron,'trainlm'); 
net5=feedforwardnet(neuron,'trainlm'); 
net6=feedforwardnet(neuron,'trainlm'); 
net7=feedforwardnet(neuron,'trainlm'); 
net8=feedforwardnet(neuron,'trainbr'); 
net1.performParam.regularization = 0;
net2.performParam.regularization = 0.1;
net3.performParam.regularization = 0.01;
net4.performParam.regularization = 0.001;
net5.performParam.regularization = 0.0001;
net6.performParam.regularization = 0.00001;
net7.performParam.regularization = 0.000001;

% net1.trainParam.goal = error;
% net2.trainParam.goal = error;
% net3.trainParam.goal = error;
% net4.trainParam.goal = error;
     
% net2=feedforwardnet(50,'trainbr');  different method
net2.iw{1,1}=net1.iw{1,1};  %set the same weights and biases for the networks 
net2.lw{2,1}=net1.lw{2,1};
net2.b{1}=net1.b{1};
net2.b{2}=net1.b{2};
net3.iw{1,1}=net1.iw{1,1};  %set the same weights and biases for the networks 
net3.lw{2,1}=net1.lw{2,1};
net3.b{1}=net1.b{1};
net3.b{2}=net1.b{2};
net4.iw{1,1}=net1.iw{1,1};  %set the same weights and biases for the networks 
net4.lw{2,1}=net1.lw{2,1};
net4.b{1}=net1.b{1};
net4.b{2}=net1.b{2};
net5.iw{1,1}=net1.iw{1,1};  %set the same weights and biases for the networks 
net5.lw{2,1}=net1.lw{2,1};
net5.b{1}=net1.b{1};
net5.b{2}=net1.b{2};
net6.iw{1,1}=net1.iw{1,1};  %set the same weights and biases for the networks 
net6.lw{2,1}=net1.lw{2,1};
net6.b{1}=net1.b{1};
net6.b{2}=net1.b{2};
net7.iw{1,1}=net1.iw{1,1};  %set the same weights and biases for the networks 
net7.lw{2,1}=net1.lw{2,1};
net7.b{1}=net1.b{1};
net7.b{2}=net1.b{2};
net8.iw{1,1}=net1.iw{1,1};  %set the same weights and biases for the networks 
net8.lw{2,1}=net1.lw{2,1};
net8.b{1}=net1.b{1};
net8.b{2}=net1.b{2};

net1.trainParam.epochs=training_time;
net2.trainParam.epochs=training_time;
net3.trainParam.epochs=training_time;
net4.trainParam.epochs=training_time;
net5.trainParam.epochs=training_time;
net6.trainParam.epochs=training_time;
net7.trainParam.epochs=training_time;
net8.trainParam.epochs=training_time;
%%%%aply four net



net4=train(net4,p,t);
tic; 
net1=train(net1,p,t);
time1=toc;

%second neuron
tic;
net2=train(net2,p,t);
time2 = toc;

%third neuron
tic;
net3=train(net3,p,t);
time3 = toc;

%fourth neuron
tic;
net4=train(net4,p,t);
time4 = toc;

tic;
net5=train(net5,p,t);
time5 = toc;
tic;
net6=train(net6,p,t);
time6 = toc;
tic;
net7=train(net7,p,t);
time7 = toc;
tic;
net8=train(net8,p,t);
time8 = toc;

a11=sim(net1,p); a12=sim(net2,p);
a13=sim(net3,p); a14=sim(net4,p);
a15=sim(net5,p); a16=sim(net6,p);
a17=sim(net7,p); a18=sim(net8,p);

%plots
% figure
% subplot(3,3,1);
% plot(x,y,'bx',x,cell2mat(a11),'r',x,cell2mat(a21),'g'); % plot the sine function and the output of the networks
% title('1 epoch');
% legend('target','trainlm','trainbfg','Location','north');
% subplot(3,3,2);
% postregm(cell2mat(a11),y); % perform a linear regression analysis and plot the result
% subplot(3,3,3);
% postregm(cell2mat(a21),y);
% %
% subplot(3,3,4);
% % plot(x(1:30),y(1:30),'bx',x(1:30),cell2mat(a12(1:30)),'r',x,cell2mat(a22(1:30)),'g');
% plot(x,y,'bx',x,cell2mat(a12),'r',x,cell2mat(a22),'g');
% title('50 epochs');
% legend('target','trainlm','trainbfg','Location','north');
% subplot(3,3,5);
% postregm(cell2mat(a12),y);
% subplot(3,3,6);
% postregm(cell2mat(a22),y);
% %
% subplot(3,3,7);
% plot(x,y,'bx',x,cell2mat(a13),'r',x,cell2mat(a23),'g');
% title('1000 epochs');
% legend('target','trainlm','trainbfg','Location','north');
% subplot(3,3,8);
% postregm(cell2mat(a13),y);
% subplot(3,3,9);
% postregm(cell2mat(a23),y);
%
%record the data
perfs = zeros(1, 16);
perfs(1,1) = time1;perfs(1,2) = mse(net1,t,a11);

perfs(1,3) =time2;perfs(1,4) = mse(net2,t,a12);

perfs(1,5) =time3;perfs(1,6) = mse(net3,t,a13);

perfs(1,7) =time4;perfs(1,8) = mse(net4,t,a14);

perfs(1,9) = time5;perfs(1,10) = mse(net5,t,a15);

perfs(1,11) =time6;perfs(1,12) = mse(net6,t,a16);

perfs(1,13) =time7;perfs(1,14) = mse(net7,t,a17);

perfs(1,15) =time8;perfs(1,16) = mse(net8,t,a18);
%%%%%%%%%%%%%%%%%
figure
end_point = 135;
% subplot(1,3,7);
plot(x(1:end_point),y(1:end_point),'bx');
hold on
plot(x(1:end_point),cell2mat(a11(1:end_point)),'r',x(1:end_point),cell2mat(a12(1:end_point)),'g',x(1:end_point),cell2mat(a13(1:end_point)),'c',x(1:end_point),cell2mat(a14(1:end_point)),'y')
hold on
plot(x(1:end_point),cell2mat(a15(1:end_point)),'color',[0.8 0.2 0])
hold on
plot(x(1:end_point),cell2mat(a16(1:end_point)),'color',[0.2 0.1 0.9]);
hold on
plot(x(1:end_point),cell2mat(a17(1:end_point)),'color',[0.4 0.2 0.1]);
hold on
plot(x(1:end_point),cell2mat(a18(1:end_point)),'color',[0.0 0.3 0.4]);
title('trainlm and trainbr');
legend('target','0','0.1','0.01','0.001','0.0001','0.00001','0.000001','trainbr','Location','north');


