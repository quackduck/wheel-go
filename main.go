package main

import "fmt"

func main() {

	n := network{
		newRandomLayer(0, 1), // input layer
		newRandomLayer(1, 4),
		newRandomLayer(4, 4),
		newRandomLayer(4, 1),
	}

	//fmt.Println(n[1].weights)

	input := []float64{1}
	fmt.Println(n.forward(input))

	// backpropagation
	for i := 0; i < 100_000; i++ {
		for d := 0.0; d < 1; d += 0.01 {
			input := []float64{d}
			target := []float64{d * d}
			n.forward(input)
			n.backward(target)
		}

		//input := []float64{0.5}
		//target := []float64{0.5}
		//n.forward(input)
		//n.backward(target)
		//
		//input = []float64{0.6}
		//target = []float64{0.1}
		//n.forward(input)
		//n.backward(target)
	}

	for d := 0.0; d < 1; d += 0.01 {
		input := []float64{d}
		fmt.Println(input, n.forward(input)[0])
		//fmt.Println(n.forward(input)[0])
	}

	//fmt.Println(n[1].weights)
}
