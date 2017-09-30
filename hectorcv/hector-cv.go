package main

import (
	"fmt"
	"log"
	"os"
	"runtime"
	"runtime/pprof"
	"strconv"

	"github.com/xlvector/hector"
	"github.com/xlvector/hector/core"
	"github.com/pantsing/hector/eval"
)

func SplitFile(dataset *core.DataSet, total, part int) (*core.DataSet, *core.DataSet) {

	train := core.NewDataSet()
	test := core.NewDataSet()

	for i, sample := range dataset.Samples {
		if i%total == part {
			test.AddSample(sample)
		} else {
			train.AddSample(sample)
		}
	}
	return train, test
}

func main() {
	train_path, _, _, method, params := hector.PrepareParams()
	global, _ := strconv.ParseInt(params["global"], 10, 64)
	profile, _ := params["profile"]
	dataset := core.NewDataSet()
	dataset.Load(train_path, global)
	runtime.GOMAXPROCS(runtime.NumCPU())
	cv, _ := strconv.ParseInt(params["cv"], 10, 32)
	total := int(cv)

	if profile != "" {
		fmt.Println(profile)
		f, err := os.Create(profile)
		if err != nil {
			fmt.Println("%v", err)
			log.Fatal(err)
		}
		pprof.StartCPUProfile(f)
		defer pprof.StopCPUProfile()
	}

	average_auc := 0.0
	average_er := 0.0
	for part := 0; part < total; part++ {
		train, test := SplitFile(dataset, total, part)
		classifier := hector.GetClassifier(method)
		classifier.Init(params)
		auc, predictions := hector.AlgorithmRunOnDataSet(classifier, train, test, "", params)
		fmt.Printf("AUC: %.20g\n", auc)
		average_auc += auc
		er :=eval.ErrorRate(predictions)
		fmt.Printf("ER: %.20g\n", er)
		average_er += er
		classifier = nil
	}
	fmt.Println(average_auc / float64(total))
	fmt.Println(average_er / float64(total))
}
