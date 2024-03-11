package main

import (
	"fmt"
	"math"
	"strings"
)

func analyze(n *network) {
	for i, l := range n.Layers {
		if i == 0 {
			continue
		}
		fmt.Println("Layer", i)
		fmt.Println("Weights")
		for i, w := range l.Weights {
			fmt.Println("Neuron", i+1, "Weights")
			histogram(w)
		}
		fmt.Println("Biases")
		histogram(l.Biases)
	}
}

func histogram(data []float64) {
	n := len(data)
	min := data[0]
	max := data[0]
	for _, v := range data {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	bins := int(math.Sqrt(float64(n)))
	binSize := (max - min) / float64(bins)
	counts := make([]int, bins)
	for _, v := range data {
		bin := int((v - min) / binSize)
		if bin == bins {
			bin-- // the max value is in the last bin
		}
		counts[bin]++
	}
	for i, c := range counts {
		fmt.Printf("%5.2f - %5.2f | %s\n", min+float64(i)*binSize, min+float64(i+1)*binSize, strings.Repeat("*", c))
	}
}
