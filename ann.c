/* Copyright (C) 2020 David Brunecz. Subject to GPL 2.0 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define DIM(x)	(sizeof(x)/sizeof((x)[0]))

/******************************************************************************/

float frand(void)
{
	float f;

	f = 2.0f * rand() / (float)RAND_MAX;
	return -1.0f + f;
}

float slope(float x)
{
	return x * (1.0f - x);
}

float sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

float mu;
float err;

#if 1
#define LAYERS 3
int layers[LAYERS] = { 2, 2, 1 };
#else
#define LAYERS 4
int layers[LAYERS] = { 2, 2, 2, 1 };
#endif

int w_idxs[LAYERS];
int o_idxs[LAYERS];
int t_idxs[LAYERS];
float *arr;

#define TMPDX(l, n, p)		arr[t_idxs[l] + n * layers[l + 1] + p]
#define WEIGHT(l, n, w)		arr[w_idxs[l] + n * (layers[l - 1] + 1) + w]
#define OUTPUT(l, n)		arr[o_idxs[l] + n]

void ann_init(void)
{
	int i, j, k, sz;

	j = 0;
	for (i = 0; i < DIM(layers); i++) {
		o_idxs[i] = j;
		j += layers[i];
	}

	for (i = 1; i < DIM(layers); i++) {
		sz = layers[i - 1] + 1; /* weights + bias */
		sz *= layers[i]; /* per node */

		w_idxs[i] = j;
		j += sz;
	}

	for (i = 1; i < DIM(layers) - 1; i++) {
		sz = layers[i];
		sz *= layers[i + 1];
		t_idxs[i] = j;
		j += sz;
	}

	arr = malloc(sizeof(*arr) * j);

	for (i = 1; i < DIM(layers); i++)
		for (j = 0; j < layers[i]; j++)
			for (k = 0; k < layers[i - 1] + 1; k++)
				WEIGHT(i, j, k) = frand();
}

void display(void)
{
	int layer, node, weight;

	printf("i ");
	for (node = 0; node < layers[0]; node++)
		printf("%2.2f ", OUTPUT(0, node));
	printf("  ");

	for (layer = 1; layer < DIM(layers); layer++) {
		printf(" | l%d ", layer);
		for (node = 0; node < layers[layer]; node++) {
			printf(" n%d o%2.2f w ", node, OUTPUT(layer, node));
			for (weight = 0; weight < layers[layer - 1]; weight++) {
				printf("%2.2f ", WEIGHT(layer, node, weight));
			}
		}
	}
	printf("\n");
}

void fprop(void)
{
	int i, j, k;

	for (i = 1; i < DIM(layers); i++) {
		for (j = 0; j < layers[i]; j++) {
			OUTPUT(i, j) = 0;
			for (k = 0; k < layers[i - 1]; k++)
				OUTPUT(i, j) += WEIGHT(i, j, k) * OUTPUT(i - 1, k);
			OUTPUT(i, j) += WEIGHT(i, j, k) * 1.0f;

			OUTPUT(i, j) = sigmoid(OUTPUT(i, j));
		}
	}
}

void bprop(void)
{
	int layer, node, parent, weight;
	int parent_layers;
	float slp, dx;

	for (layer = DIM(layers) - 1; layer; layer--) {
		for (node = 0; node < layers[layer]; node++) {
			parent_layers = (layer == (DIM(layers) - 1)) ?
							1 : layers[layer + 1];
			for (parent = 0; parent < parent_layers; parent++) {
				slp = layer < (DIM(layers) - 1) ?
					TMPDX(layer, node, parent) : 1.0f;
				slp *= slope(OUTPUT(layer, node));
				for (weight = 0; weight < layers[layer - 1] + 1; weight++) {
					dx = slp * OUTPUT(layer - 1, weight);
					if (layer > 1)
						TMPDX(layer - 1, weight, node) = dx;
					WEIGHT(layer, node, weight) += dx * err * mu;
				}
			}
		}
	}
}

int main(int argc, char *argv[])
{
	int inp_idx = o_idxs[0];
	float v, o;
	int i;

	ann_init();

	mu = 2.6f;

	for (i = 0; i < 8000; i++) {
		arr[inp_idx + 0] = i & 1 ? 1.0f : 0.0f;
		arr[inp_idx + 1] = i & 2 ? 1.0f : 0.0f;
		v = (((i & 3) == 1) || ((i & 3) == 2)) ? 1.0f : 0.0f;

		fprop();

		o = OUTPUT(DIM(layers) - 1, 0);
		err = v - o;

		if (i && (i % 50) < 4)
			printf("%2.2f %2.2f | %2.2f - %2.2f = err % 2.2f\n",
				arr[inp_idx + 0], arr[inp_idx + 1], v, o, err);

		bprop();
	}

	return 0;
}

