package main

import (
	"fmt"
	"github.com/petar/GoMNIST"
	"math/rand/v2"
)

var trainData, testData *GoMNIST.Set
var indices []int

func init() {
	var err error
	trainData, testData, err = GoMNIST.Load("./data")
	if err != nil {
		panic(err)
	}
	indices = rand.Perm(60000) // shuffle
}

var currIndex int

func makeTrainingDataMNIST() (batch []pair) {
	size := 1000 // batch size
	//size := 20 // batch size

	batch = make([]pair, size)

	for i := 0; i < size; i++ {
		img, label := trainData.Get(indices[(i+currIndex)%60000]) // random order
		batch[i].input = make([]float64, 28*28)
		batch[i].output = make([]float64, 10)
		for j := 0; j < 28*28; j++ {
			batch[i].input[j] = float64(img[j]) / 255 // normalize
		}

		for j := 0; j < 10; j++ {
			batch[i].output[j] = -1
		}
		batch[i].output[label] = 1
	}
	currIndex += size
	if currIndex == 60000 {
		fmt.Println("Epoch finished")
		indices = rand.Perm(60000) // shuffle
	}
	currIndex %= 60000 // this is optional because the index wraps around anyway
	return
}

func makeTrainingDataMNISTRecreate() (batch []pair) {
	size := 1000 // batch size

	batch = make([]pair, size)

	for i := 0; i < size; i++ {
		img, _ := trainData.Get(indices[(i+currIndex)%60000]) // random order
		batch[i].input = make([]float64, 28*28)
		for j := 0; j < 28*28; j++ {
			batch[i].input[j] = float64(img[j]) / 255 // normalize
		}
		batch[i].output = batch[i].input // recreate the input
	}
	currIndex += size
	if currIndex == 60000 {
		fmt.Println("Epoch finished")
		indices = rand.Perm(60000) // shuffle
	}
	currIndex %= 60000 // this is optional because the index wraps around anyway
	return
}

var imagesByLabel [][][]float64

func init() {
	imagesByLabel = make([][][]float64, 10)
	for i := 0; i < 10; i++ {
		imagesByLabel[i] = make([][]float64, 0, 6000)
	}
	for i := 0; i < 60000; i++ {
		img, label := trainData.Get(i)
		image := make([]float64, 28*28)
		for j := 0; j < 28*28; j++ {
			image[j] = float64(img[j]) / 255 // normalize
		}
		imagesByLabel[label] = append(imagesByLabel[label], image)
	}
}

func getRandomImageByLabel(label int) []float64 {
	return imagesByLabel[label][rand.IntN(len(imagesByLabel[label]))]
}

func makeTrainingDataMNISTDevelopHandwriting() (batch []pair) {
	// for every input image, the target is another image of the same label
	size := 1000 // batch size

	batch = make([]pair, size)

	for i := 0; i < size; i++ {
		img, label := trainData.Get(indices[(i+currIndex)%60000]) // random order
		batch[i].input = make([]float64, 28*28)
		for j := 0; j < 28*28; j++ {
			batch[i].input[j] = float64(img[j]) / 255 // normalize
		}
		batch[i].output = getRandomImageByLabel(int(label)) // get a random image of the same label
	}
	currIndex += size
	if currIndex == 60000 {
		fmt.Println("Epoch finished")
		indices = rand.Perm(60000) // shuffle
	}
	currIndex %= 60000 // this is optional because the index wraps around anyway
	return
}

func makeTrainingDataMNISTDevelopHandwriting2() (batch []pair) {
	// for every input image, the target is another image of the same label
	size := 1000 // batch size

	batch = make([]pair, size)

	for i := 0; i < size; i++ {
		img, label := trainData.Get(indices[(i+currIndex)%60000]) // random order
		batch[i].output = make([]float64, 28*28)
		for j := 0; j < 28*28; j++ {
			batch[i].output[j] = float64(img[j]) / 255 // normalize
		}
		batch[i].input = make([]float64, 10)
		for j := 0; j < 10; j++ {
			batch[i].input[j] = -1
		}
		batch[i].input[label] = 1
	}
	currIndex += size
	if currIndex == 60000 {
		fmt.Println("Epoch finished")
		indices = rand.Perm(60000) // shuffle
	}
	currIndex %= 60000 // this is optional because the index wraps around anyway
	return
}

func makeTrainingDataMNISTAddOne() (batch []pair) {
	// for every input image, the target is another image of the same label
	size := 1000 // batch size

	batch = make([]pair, size)

	for i := 0; i < size; i++ {
		img, label := trainData.Get(indices[(i+currIndex)%60000]) // random order
		batch[i].input = make([]float64, 28*28)
		for j := 0; j < 28*28; j++ {
			batch[i].input[j] = float64(img[j]) / 255 // normalize
		}
		//batch[i].output = getRandomImageByLabel(int(label+1) % 10)
		batch[i].output = imagesByLabel[(label+1)%10][0] // get the first image of the next label
	}
	currIndex += size
	if currIndex == 60000 {
		fmt.Println("Epoch finished")
		indices = rand.Perm(60000) // shuffle
	}
	currIndex %= 60000 // this is optional because the index wraps around anyway
	return
}

func makeTestingDataMNIST() (inputs, targets [][]float64) {
	size := 1000

	inputs = make([][]float64, size)
	targets = make([][]float64, size)
	for i := 0; i < size; i++ {
		img, label := testData.Get(i)
		//fmt.Println(len(img), label)
		inputs[i] = make([]float64, 28*28)
		targets[i] = make([]float64, 10)
		for j := 0; j < 28*28; j++ {
			inputs[i][j] = float64(img[j]) / 255 // normalize
		}

		for j := 0; j < 10; j++ {
			targets[i][j] = -1
		}
		targets[i][label] = 1
	}
	return
}
