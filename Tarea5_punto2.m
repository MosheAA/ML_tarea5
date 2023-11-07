clc, clear all, close all 

%% Punto 2 
%implementacion del clasificador gaussiano en las funciones:
%clasificador_gaussiano.m 
%clasificador_gaussiano_train.m 

%% b visualización de los datos en 2D 
label = readtable('breastCancerLabel.csv');
data = readtable('breastCancerX.csv');

T = [data,label];


gscatter(T.radius_mean,T.smoothness_mean,T.Cancer)
xlabel('radius mean')
ylabel('smoothness mean')


%% c Entrenamiento y clasificación 

% probabilidad a priori de malignidad 
PM=0.017;

%seleccion de caracteristicas para clasificar 
v = {'radius_mean' 'smoothness_mean'};


new_data_pre = data{:,v};

% Estandarización de los datos 
new_data = (new_data_pre-mean(new_data_pre))./var(new_data_pre);
label_total = label{:,:};

% Selección datos de entrenamiento 
M =find(label_total==1);
N =find(label_total==0);
train_size = 150;

data_M_E= new_data(M(1:train_size),:); 
data_N_E=new_data(N(1:train_size),:);

x = cat(1,data_M_E,data_N_E); 
label = cat(1, label_total(M(1:train_size),:),label_total(N(1:train_size),:));

% Entrenamiento 
gauss_model = clasificador_gaussiano_train(x, label );


% Selección datos de prueba 
data_M_T= new_data(M(train_size+1:end),:); 
data_N_T=new_data(N(train_size+1:end),:); 

data_Test = cat(1,data_M_T,data_N_T);

% Clasificación datos de prueba 
label_pred = clasificador_gaussiano(data_Test, gauss_model, PM );



%% d Visualizacion frontera del clasificador 

% Rangos de las dos caracteristicas utilizadas en c. 
a_x1 = min(new_data(:,1));
b_x1 = max(new_data(:,1));

a_x2 = min(new_data(:,2));
b_x2 = max(new_data(:,2));

% Generacion datos aleatorios en R2 
testdata = [(b_x1-a_x1).*rand(1000,1) + a_x1 , (b_x2-a_x2).*rand(1000,1) + a_x2];

% Clasificación datos aleatorios 
label_pred_d = clasificador_gaussiano(testdata, gauss_model, PM);

% Grafica frontera 
gscatter(testdata(:,1),testdata(:,2),label_pred_d,[],'x')

xlabel('radius mean')
ylabel('smoothness mean')
legend('Location','northeastoutside')
hold on

% Grafica predicciones set de prueba 
gscatter(data_Test(:,1),data_Test(:,2),label_pred)

hold off