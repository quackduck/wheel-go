package main

import "fmt"

func main() {
	input := newRandomLayer(1, 4)
	//fmt.Println(layer1.weights)
	layer2 := newRandomLayer(4, 4)
	//layer3 := newRandomLayer(4, 4)
	out := newRandomLayer(4, 1)

	//layer1.connectNext(layer2)
	//layer2.connectNext(layer3)
	//layer3.connectNext(out)
	//out.connectPrev(layer2)

	input.connectNext(layer2)
	layer2.connectNext(out)
	//layer3.connectNext(out)
	out.connectPrev(layer2)
	layer2.connectPrev(input)

	//layer1.connectNext(out)
	//out.connectPrev(layer1)
	// forward pass
	input.forward([]float64{1})
	// print the activations of the last layer
	fmt.Println(out.activations)

	for n := 0; n < 10000000; n++ { // train
		//for i := 0.0; i <= 1; i += 0.1 { // learn the function f(x) = x^2
		//	layer1.forward([]float64{i})
		//	out.backward([]float64{i * i})
		//
		//	//layer1.forward([]float64{0})
		//	//out.backward([]float64{1})
		//}
		input.forward([]float64{1})
		out.backward([]float64{0})
		input.forward([]float64{0})
		out.backward([]float64{1})
	}

	// now forward pass again
	input.forward([]float64{1.0})
	fmt.Println(out.activations)

	for i := 0.0; i < 1.0; i += 0.1 {
		input.forward([]float64{i})
		fmt.Println(i, out.activations)
	}

	input.forward([]float64{0})
	fmt.Println(out.activations)

	// print all the weights

	//fmt.Println(layer1.weights)
	//fmt.Println(layer2.weights)
	//fmt.Println(out.weights)

	// print every 100th cost
	//for i := 0; i < len(costList); i += 100 {
	//	fmt.Println(costList[i])
	//}

}
