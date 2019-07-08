%%0.1 read data

book_fname = 'goblet_book.txt';
fid = fopen(book_fname,'r');
book_data = fscanf(fid,'%c');
fclose(fid);

book_chars = unique(book_data);

K = length(book_chars);

char_to_ind = containers.Map('KeyType','char','ValueType','int32');
ind_to_char = containers.Map('KeyType','int32','ValueType','char');

for i= 1:K
    value = book_chars(i);
    char_to_ind(value)= i;
    ind_to_char(i)= value;
end

% 0.2 set hyper param & init the RNN param

m = 100; % hidden state
sig = 0.05;
eta = 0.1;
seq_length = 25;
epochs = 1;

eps = 0.00001;

[RNN, m_teta] = param(K,m,sig); 

%% compare gradients

X_chars = book_data(1:seq_length);
Y_chars = book_data(2:seq_length+1);

X = char_to_oneHot(X_chars,char_to_ind,K); %give forward only onehot
Y = char_to_oneHot(Y_chars,char_to_ind,K);

h0 = zeros(m,1); 

[grads] = backward(RNN, h0, X, Y);

h = 1e-4;

[num_grads] = ComputeGradsNum(X, Y, RNN, h);


for i = 1:size(grads.W,2)
    sumsW(i) = sqrt(sum((grads.W(:,i)-num_grads.W(:,i)).^2));
end

compare_gradW = sum(sumsW)

for i = 1:size(grads.V,2)
    sumsV(i) = sqrt(sum((grads.V(:,i)-num_grads.V(:,i)).^2));
end

compare_gradV = sum(sumsV)

for i = 1:size(grads.U,2)
    sumsU(i) = sqrt(sum((grads.U(:,i)-num_grads.U(:,i)).^2));
end

compare_gradU = sum(sumsU)
  
compare_gradb = sqrt(sum((grads.b-num_grads.b).^2))

compare_gradc = sqrt(sum((grads.c-num_grads.c).^2))

%% 0.5 AdaGrad

iter = 0;

for i = 1:epochs

e = 1; %keeps track where in the book you are

hprev =  zeros(m,1);

while  length(book_data)-seq_length-1 > e
    
X_chars = book_data(e:e+seq_length-1);
Y_chars = book_data(e+1:e+seq_length);
    
X = char_to_oneHot(X_chars,char_to_ind,K);
Y = char_to_oneHot(Y_chars,char_to_ind,K);

[~,H,loss,~,~] = forward(RNN, hprev, X, Y);
hprev = H(:,end);

[grads] = backward(RNN, hprev, X, Y);


for f = fieldnames(RNN)'
    
    m_teta.(f{1}) = m_teta.(f{1})+grads.(f{1}).^2;    

    RNN.(f{1}) = RNN.(f{1}) - eta*grads.(f{1})./(sqrt(m_teta.(f{1})+eps)) ;
end


iter = iter +1;

if iter == 1
    smooth_loss = loss;
else
    smooth_loss = 0.999*smooth_loss + 0.001*loss;
end


loss_list(iter)= smooth_loss;

e = e + seq_length;
end

end
%% plot loss

plot(loss_list)
ylabel('loss function');
xlabel('iter');

%% Generate text


iter = 0;
n = 200;

for i = 1:epochs

e = 1; %keeps track where in the book you are

hprev =  zeros(m,1);

while  length(book_data)-seq_length-1 > e
    
X_chars = book_data(e:e+seq_length-1);
Y_chars = book_data(e+1:e+seq_length);
    
X = char_to_oneHot(X_chars,char_to_ind,K);
Y = char_to_oneHot(Y_chars,char_to_ind,K);

[~,H,loss,~,~] = forward(RNN, hprev, X, Y);
hprev = H(:,end);

[grads] = backward(RNN, hprev, X, Y);


for f = fieldnames(RNN)'
    
    m_teta.(f{1}) = m_teta.(f{1})+grads.(f{1}).^2;    

    RNN.(f{1}) = RNN.(f{1}) - eta*grads.(f{1})./(sqrt(m_teta.(f{1})+eps)) ;
end



iter = iter +1;

if iter == 1
    smooth_loss = loss;
else
    smooth_loss = 0.999*smooth_loss + 0.001*loss;
end


if iter == 1
    iter
    
    smooth_loss
 
    oneHot_to_char(synt_a_seq_of_char(RNN, hprev, X(:,1), n, K),ind_to_char)
end

if mod(iter,10000) == 0
    iter
    
    smooth_loss
    
    oneHot_to_char(synt_a_seq_of_char(RNN, hprev, X(:,1), n, K),ind_to_char)
end

loss_list(iter)= smooth_loss;

e = e + seq_length;
end



end

%% best model and generate 1000 chars

allTexts = {};
allLoss = [];
m = 100;
sig = 0.05;
e_min= -4;
e_max= -1;

for i = 1:40
    
e = e_min + (e_max - e_min)*rand(1, 1);
eta(i) = 10^e;

RNN.V = randn(K,m)*sig;
RNN.W = randn(m,m)*sig;
RNN.U = randn(m,K)*sig;
RNN.c = zeros(K,1);
RNN.b = zeros(m,1);

m_teta.V = zeros(K,m);
m_teta.W = zeros(m,m);
m_teta.U = zeros(m,K);
m_teta.c = zeros(K,1);
m_teta.b = zeros(m,1);

[loss, text] = find_best_model(book_data, RNN,eta(i), m_teta, char_to_ind, ind_to_char);

allTexts{i} = text;
allLoss(i) = loss;

end

[MinLoss,Index] = min(allLoss);
Besteta = eta(Index);
Besttext = allTexts{Index};

%% functions
function [RNN, m_teta] = param(K,m,sig) 
RNN.V = randn(K,m)*sig;
RNN.W = randn(m,m)*sig;
RNN.U = randn(m,K)*sig;
RNN.c = zeros(K,1);
RNN.b = zeros(m,1);

m_teta.V = zeros(K,m);
m_teta.W = zeros(m,m);
m_teta.U = zeros(m,K);
m_teta.c = zeros(K,1);
m_teta.b = zeros(m,1);

end

function [out] = synt_a_seq_of_char(RNN, h, x, n, K)

out = zeros(K,n);
for i = 1:n

a = RNN.W*h+RNN.U*x + RNN.b;

h = tanh(a);
o = RNN.V*h+ RNN.c;
p = softmax(o);
% generate next input x_t+1
cp = cumsum(p);
a = rand;
ixs = find(cp-a >0);
aa = ixs(1);
out(aa,i)=1;
x=out(:,i);

end

end

function [Y] = char_to_oneHot(X, char_to_ind,K)

Y = zeros(K, size(X,2)); 
for i = 1:size(X,2)
key = char_to_ind(X(i));
Y(key, i) = 1;    

end
end

function [Y] = oneHot_to_char(X, ind_to_char)

Y = [];

for i = 1:size(X,2)
    [key,~] = find(X(:,i));
    Y=[Y, ind_to_char(key)];

end
end

function [p, H, L, o, a] = forward(RNN, h0, X, Y)

[m,n] = size(X);
H = zeros(size(h0,1),n);

p = zeros(m,n);
o =  zeros(m,n);
h = h0;
a = zeros(size(h0,1),n);

for i = 1:n
x = X(:,i);


a(:,i) = RNN.W*h+RNN.U*x + RNN.b;

h = tanh(a(:,i));
H(:,i) = h;
o(:,i) = RNN.V*h+ RNN.c;
p(:,i) = softmax(o(:,i));

loss(i) = -log((Y(:,i))'*p(:,i));

end

L = sum(loss);

end


function [grads] = backward(RNN, h0, X, Y)

[P, H,~ , ~, a] = forward(RNN, h0, X, Y);

grads.V=zeros(size(RNN.V));
grads.W=zeros(size(RNN.W));
grads.U=zeros(size(RNN.U));

G = -(Y-P)';
grads.c = (sum(G,1))';

grads.V = grads.V + G'*H';

g_tau = G(end,:);
a_tau = a(:,end);

dldh_tau = g_tau*RNN.V;
dlda_tau = dldh_tau*diag(1-(tanh(a_tau)).^2);

allVectors_dldat = zeros(size(X,2), size(h0,1));

allVectors_dldat(end,:) = dlda_tau;

dlda_i = dlda_tau; 

for i = size(X,2)-1:-1:1
    g_i = G(i,:);
    dldh_i = g_i*RNN.V+dlda_i*RNN.W;
    a_i = a(:,i);
    dlda_i = dldh_i*diag(1-(tanh(a_i)).^2);
    allVectors_dldat(i,:) = dlda_i;
end

%to get h_t-1
H(:,end) = [];
H = [h0 H];

grads.W = grads.W + allVectors_dldat'*H';
grads.U = grads.U + allVectors_dldat'*X';
grads.b = (sum(allVectors_dldat,1))';

for f = fieldnames(grads)'
     grads.(f{1})=max(min(grads.(f{1}), 5), -5);
end

end

function [L] = ComputeLoss(X, Y, RNN, h0)

[P,~,~,~,~] = forward(RNN, h0, X, Y);

[~,n] = size(P);

for i = 1:n

    loss(i) = -log((Y(:,i))'*P(:,i));
    
end

L = sum(loss);


end

function [smooth_loss, text] = find_best_model(book_data, RNN,eta, m_teta,char_to_ind, ind_to_char)
seq_length = 25;
eps = 0.00001;
e = 1;
m = 100; % hidden state
K = 83;
iter = 0;
hprev =  zeros(m,1);

while  length(book_data)-seq_length-1 > e
    
X_chars = book_data(e:e+seq_length-1);
Y_chars = book_data(e+1:e+seq_length);
    
X = char_to_oneHot(X_chars,char_to_ind,K);
Y = char_to_oneHot(Y_chars,char_to_ind,K);

[~,H,loss,~,~] = forward(RNN, hprev, X, Y);
hprev = H(:,end);

[grads] = backward(RNN, hprev, X, Y);


for f = fieldnames(RNN)'
    
    m_teta.(f{1}) = m_teta.(f{1})+grads.(f{1}).^2;    

    RNN.(f{1}) = RNN.(f{1}) - eta*grads.(f{1})./(sqrt(m_teta.(f{1})+eps)) ;
end

iter = iter + 1;

if iter == 1
    smooth_loss = loss;
else
    smooth_loss = 0.999*smooth_loss + 0.001*loss;
end

e = e + seq_length;
end

text = oneHot_to_char(synt_a_seq_of_char(RNN, hprev, X(:,1), 1000, K),ind_to_char);

end
