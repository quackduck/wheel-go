package main

import (
	"fmt"
	"math"
	"math/rand/v2"
)

type layer struct {
	// weights[i][j] is the weight from neuron i of this layer to neuron j of the previous layer
	// therefore len(weights) is the number of neurons in this layer and len(weights[0]) is the number of inputs
	weights     [][]float64 // weights for each neuron
	biases      []float64   // bias for each neuron
	activations []float64   // activations for each neuron
	zs          []float64   // weighted sum for each neuron + bias
	//input       []float64
	//nextLayer *layer
	//prevLayer *layer
}

func newLayer(weights [][]float64, biases []float64) *layer {
	return &layer{
		weights:     weights,
		biases:      biases,
		activations: make([]float64, len(weights)),
		zs:          make([]float64, len(weights)),
	}
}

func newRandomLayer(inputSize, thisSize int) *layer {
	return newLayer(randomWeights(inputSize, thisSize), randomBiases(thisSize))
}

func randomWeights(inputSize, thisSize int) [][]float64 {
	weights := make([][]float64, thisSize)
	for i := range weights {
		weights[i] = make([]float64, inputSize)
		for j := range weights[i] {
			weights[i][j] = rand.Float64()*2 - 1
			//weights[i][j] = rand.NormFloat64()
		}
	}
	return weights
}

func randomBiases(size int) []float64 {
	biases := make([]float64, size)
	for i := range biases {
		//biases[i] = (rand.Float64()*2 - 1) * 0
		biases[i] = 0
	}
	return biases
}

//func activate(x float64) float64 {
//	return 1 / (1 + math.Exp(-x))
//}
//
//func activatePrime(x float64) float64 {
//	return activate(x) * (1 - activate(x))
//}

// try tanh()
func activate(x float64) float64 {
	return math.Tanh(x)
}

func activatePrime(x float64) float64 {
	return 1 - math.Tanh(x)*math.Tanh(x)
}

func costFunc(err float64) float64 {
	return err * err
}

func costPrime(err float64) float64 {
	return 2 * err
}

type network struct {
	layers []*layer
	cost   float64
}

func (n *network) forward(input []float64) []float64 {
	n.layers[0].activations = input // set the input layer's activations
	for i := range n.layers {
		if i == 0 {
			continue
		}
		//fmt.Println("forwarding layer", i, "with input", n[i-1].activations, "weights", n[i].weights, "biases", n[i].biases)
		n.layers[i].forward(input)
		input = n.layers[i].activations
	}
	return n.layers[len(n.layers)-1].activations
}

func (l *layer) forward(input []float64) {
	for i, weights := range l.weights { // for each neuron in this layer
		var sum float64
		for j, weight := range weights { // for each input to that neuron
			sum += weight * input[j]
		}
		l.zs[i] = sum + l.biases[i]
		l.activations[i] = activate(l.zs[i])
	}
}

func (n *network) backward(wanted []float64) {
	n.cost = 0.0
	for i, activation := range n.layers[len(n.layers)-1].activations {
		n.cost += costFunc(activation - wanted[i])
	}

	rate := 0.001

	delta := make([]float64, len(wanted)) // delta[j] is delC / delZ[j]
	for i, activation := range n.layers[len(n.layers)-1].activations {
		delta[i] = 2 * costPrime(activation-wanted[i]) * activatePrime(n.layers[len(n.layers)-1].zs[i]) // delC / delZ
		updateLayerWithDelta(n.layers[len(n.layers)-1], n.layers[len(n.layers)-2], delta, rate)
	}

	for i := len(n.layers) - 2; i >= 1; i-- { // for each layer except the input and output
		newdelta := make([]float64, len(n.layers[i].activations)) // one for each neuron
		// go over all the neurons in the next layer and add up the impact of their delta on this layer's delta
		for j := range n.layers[i].activations { // for each neuron in this layer
			var sum float64
			for k := range n.layers[i+1].activations { // for each neuron in the next layer
				sum += n.layers[i+1].weights[k][j] * delta[k]
			}
			newdelta[j] = sum * activatePrime(n.layers[i].zs[j])
		}
		updateLayerWithDelta(n.layers[i], n.layers[i-1], newdelta, rate)
		delta = newdelta
	}
}

func updateLayerWithDelta(l *layer, lp *layer, delta []float64, learnStepRate float64) {
	for i, weights := range l.weights {
		for j := range weights {
			l.weights[i][j] -= learnStepRate * errorToDelCDelWeight(delta[i], lp.activations[j])
		}
	}
	for i := range l.biases {
		l.biases[i] -= learnStepRate * errorToDelCDelBias(delta[i])
	}
}

// delta should be the error for the neuron that this weight is for
// prevActive should be the activation of the neuron that this weight is for in the previous layer
func errorToDelCDelWeight(delta, prevActive float64) float64 {
	return delta * prevActive
}

// delta should be the error for the neuron that this bias is for
func errorToDelCDelBias(delta float64) float64 {
	return delta
}

func (n *network) String() string {
	result := ""
	for i, layer := range n.layers {
		result += "Layer " + fmt.Sprint(i) + ":\n"
		result += "Weights: " + fmt.Sprint(layer.weights) + "\n"
		result += "Biases: " + fmt.Sprint(layer.biases) + "\n\n"
	}
	return result
}
