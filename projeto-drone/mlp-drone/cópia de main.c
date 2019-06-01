#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define ENTRADAS        1
#define SAIDAS          1
#define NR_AMOSTRAS     7
#define NR_NEURON_O     3
#define CAMADAS_O       2
#define EPOCAS          1000000
#define TX_APRENDIZADO  1

                                                    //entrada/prof/sexo
float cj_treinamento[NR_AMOSTRAS][ENTRADAS+SAIDAS] = {
                                                    {0.10, 0.1230, 0.1210},
                                                    {0.12, 0.1226, 0.1213},
                                                    {0.16, 0.1223, 0.1216},
                                                    {0.21, 0.1220, 0.1220},
                                                    {0.28, 0.1216, 0.1223},
                                                    {0.36, 0.1213, 0.1226},
                                                    {0.45, 0.1210, 0.1230}
                                                };


double w_e_o_1[ENTRADAS+1][NR_NEURON_O];


double w_o_s_1[NR_NEURON_O+1][NR_NEURON_O];
double w_o_s_2[NR_NEURON_O+1][SAIDAS];

double saida_o_1[NR_NEURON_O];
double saida_o_2[NR_NEURON_O];

double saida_s[SAIDAS];

double delta_saida[SAIDAS];
double gradiente_oculta_1[NR_NEURON_O];
double gradiente_oculta_2[NR_NEURON_O];
double delta_oculta[NR_NEURON_O];


/*
Cabeçalho das funções auxiliares
*/
void inicializa_sinapses();
int gera_nr_aleatorios();
void mostrar_sinapses();
double f_sigmoid(double net);
void calcular_saidas(double entradas[ENTRADAS]);
void treinar_RNA();
double calcular_erro(double desejado, double saida);
void menu();
void calcular_delta_saida(double desejado[SAIDAS]);
void calcular_delta_oculta();
void calcular_gradiente_oculta();
void ajustar_pesos_sinapticos(double entradas[ENTRADAS]);
//void escrevePesos();


/*
Função principal
*/
int main()
{
  srand(time(NULL));

  while (1) {

    menu();

  }

  return 0;
}

void inicializa_sinapses()
{
  int i, j;

  for (i = 0; i < ENTRADAS+1; i++)
    for (j =0; j < NR_NEURON_O; j++)
      w_e_o[i][j] = gera_nr_aleatorios();

  for (i = 0; i < NR_NEURON_O+1; i++)
    for (j =0; j < SAIDAS; j++)
      w_o_s[i][j] = gera_nr_aleatorios();

}

int gera_nr_aleatorios()
{
  int numeros[2] = {-1, 1};

  return (numeros[rand() % 2]);
}

void mostrar_sinapses()
{
  int i, j;


  for (i = 0; i < ENTRADAS+1; i++) {
    for (j =0; j < NR_NEURON_O; j++)
      printf("%f ", w_e_o[i][j]);
    printf("\n");
  }


  for (i = 0; i < NR_NEURON_O+1; i++) {
    for (j =0; j < SAIDAS; j++)
      printf("%f ", w_o_s[i][j]);
    printf("\n");
  }
}

double f_sigmoid(double net)
{
  return 1 / (1 + exp(-net));
}

void calcular_saidas(double entradas[ENTRADAS])
{
  int i, j;


  for (i = 0; i < NR_NEURON_O; i++) {
    saida_o[i] = 0.0;
    saida_o[i] += w_e_o[0][i] * 1;

    for (j =1; j < ENTRADAS+1; j++)
     saida_o[i] += w_e_o[j][i] * entradas[j-1];

    saida_o[i] = f_sigmoid(saida_o[i]);
  }

  for (i = 0; i < SAIDAS; i++) {
    saida_s[i] = 0.0;
    saida_s[i] += w_o_s[0][i] * 1;

    for (j = 1; j < NR_NEURON_O+1; j++)
      saida_s[i] += w_o_s[j][i] * saida_o[j-1];

    saida_s[i] = f_sigmoid(saida_s[i]);
  }

}

void treinar_RNA(int coluna)
{
  int i, j;
  double entradas[ENTRADAS];
  double saidas[SAIDAS];

  for (i = 1; i <= EPOCAS; i++) {

    for (j = 0; j < NR_AMOSTRAS; j++) {
        entradas[0] = cj_treinamento[j][0];

        calcular_saidas(entradas);

        calcular_delta_saida(cj_treinamento[j][coluna]);
        calcular_gradiente_oculta();
        calcular_delta_oculta();
        ajustar_pesos_sinapticos(entradas);
    }

  }

}

double calcular_erro(double desejado, double saida)
{
  return desejado - saida;
}

void menu()
{
  int opcao, motor;
  double entradas[ENTRADAS];

  printf("Rede Neural Perceptron de Multiplas Camadas\n");
  printf("Problema do OU EXCLUSIVO - XOR\n");
  printf("*******************************************\n");
  printf("1.Treinar a rede\n");
  printf("2.Usar a rede\n");
  printf("3.Ver pesos sinpaticos\n");
  printf("4.Sair\n");
  printf("Opcao? ");
    scanf("%d", &opcao);

  switch (opcao) {

    case 1: inicializa_sinapses();
            printf("Qual motor você quer treinar 1 ou 2?");
            scanf("%d", &motor);

            treinar_RNA(motor);
            break;

    case 2: printf("Entrada 1: ");
            scanf("%lf", &entradas[0]);
            calcular_saidas(entradas);

            printf("Letra: %lf\n", saida_s);

            break;

    case 3: mostrar_sinapses();
            break;

    case 4: exit(0);

  }
}


void calcular_delta_saida(double desejado[SAIDAS])
{
    int i;
    for (i = 0; i < SAIDAS; i++) {
        delta_saida[i] = calcular_erro(desejado[i], saida_s[i]) * 
        (1 - saida_s[i] * saida_s[i]);
    }
}

void calcular_gradiente_oculta()
{
    int i, j;

	for (i = 0; i < SAIDAS; i++)
		for (j = 1; j < NR_NEURON_O+1; j++) 
			gradiente_oculta_1[j-1] = delta_saida[i] * w_o_s_1[j][i];

	for (i = 0; i < SAIDAS; i++)
		for (j = 1; j < NR_NEURON_O+1; j++) 
			gradiente_oculta_2[j-1] = delta_saida[i] * w_o_s_2[j][i];
}


void calcular_delta_oculta()
{
  int i;
	for (i = 0; i < NR_NEURON_O; i++)
    	delta_oculta[i] = gradiente_oculta_1[i] * saida_o[i] * (1 - saida_o_1[i]);

	for (i = 0; i < NR_NEURON_O; i++)
    	delta_oculta[i] = gradiente_oculta_2[i] * saida_o[i] * (1 - saida_o_2[i]);
}

void ajustar_pesos_sinapticos(double entradas[ENTRADAS])
{
  int i, j;


  for (i = 0; i < SAIDAS; i++) {
    w_o_s[0][i] = w_o_s[0][i] + TX_APRENDIZADO * delta_saida[i] * 1;

    for (j = 1; j < NR_NEURON_O+1; j++)
      w_o_s[j][i] = w_o_s[j][i] + TX_APRENDIZADO * delta_saida[i] * saida_o[j-1];

  }


      for (i = 0; i < NR_NEURON_O; i++) {
        w_e_o[0][i] = w_e_o[0][i] + TX_APRENDIZADO * delta_oculta[0] * 1;

        for (j = 1; j < ENTRADAS+1; j++)
         w_e_o[j][i] = w_e_o[j][i] + TX_APRENDIZADO * delta_oculta[i] * entradas[j-1];
      }

}

