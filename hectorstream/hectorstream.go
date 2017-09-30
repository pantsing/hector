package main

import (
	"fmt"
	"log"
	"runtime"

	"github.com/xlvector/hector"
	"github.com/xlvector/hector/core"
	"github.com/xlvector/hector/lr"
	"os"
	"strconv"
)

func main() {
	train, test, pred, method, params := hector.PrepareParams()
	log.SetFlags(log.LstdFlags | log.Lshortfile)

	action, _ := params["action"]
	global, _ := strconv.ParseInt(params["global"], 10, 64)

	runtime.GOMAXPROCS(runtime.NumCPU())
	if action == "train" {
		classifier := &lr.LogisticRegressionStream{}
		classifier.Init(params)
		data := core.NewStreamingDataSet()
		go data.Load(train, 1)
		classifier.Train(data)
		classifier.SaveModel(params["model"])
	} else if action == "test" {
		classifier := &lr.LogisticRegression{}
		classifier.Init(params)
		auc, _, _ := hector.AlgorithmTest(classifier, test, pred, params)
		fmt.Println("AUC:")
		fmt.Println(auc)
	} else if action == "predict" {
		model_path, _ := params["model"]
		classifier := hector.GetClassifier(method)
		classifier.Init(params)
		if model_path != "" {
			classifier.LoadModel(model_path)
		} else {
			return
		}
		data := core.NewStreamingDataSet()
		go data.LoadFromStdIn(global)
		var pred_file *os.File
		if pred_path != "" {
			pred_file, _ = os.Create(pred_path)
		}
		for sample := range data.Samples {
			prediction := classifier.Predict(sample)
			if pred_file != nil {
				pred_file.WriteString(strconv.FormatFloat(prediction, 'g', 5, 64) + "\n")
				//fmt.Println(strconv.FormatFloat(prediction, 'g', 5, 64), sample.Label, sample.Features)
				//pred_file.WriteString(fmt.Sprint(strconv.FormatFloat(prediction, 'g', 5, 64), sample.Label, sample.Features, "\n"))
			}
		}
		if pred_file != nil {
			pred_file.Close()
		}
	}
}
