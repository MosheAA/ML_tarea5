

function gauss_model = clasificador_gaussiano_train(x, label )
% Funcion para generar matriz de covarianza y de medias a partir de datos
% de entrenamiento 

% x: matriz de datos de entrenamiento con las dos clases
% label: vector con etiquetas de x 

% estructura que guarda el modelo 
gauss_model= struct;
data = x;

n= size(data,2); % dimensiones del problema 



new_T_data = data;

% Identificación de datos de cada clase 
M =find(label==1);
N =find(label==0);


data_M_E= new_T_data(M(1:end),:); 
data_N_E=new_T_data(N(1:end),:); 

% Estimación de vectores de medias y matrices de covarianza para cada clase
mu_x_M = 1/size(data_M_E,1)*sum(data_M_E);
D_M=data_M_E-mu_x_M;

C_M= 0;
Q_M= D_M'*D_M./(size(data_M_E,1)-1) + C_M.*eye(n);

mu_x_N = 1/size(data_N_E,1)*sum(data_N_E);
D_N=data_N_E-mu_x_N;

C_N= 0;
Q_N= D_N'*D_N./(size(data_N_E,1)-1) + C_N*eye(n);

% Guardar modelo 
gauss_model.QC1 = Q_M;
gauss_model.MUC1 = mu_x_M;

gauss_model.QC2 = Q_N;
gauss_model.MUC2 = mu_x_N;



end