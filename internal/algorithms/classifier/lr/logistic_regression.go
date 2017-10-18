package lr

import (
	"bufio"
	"github.com/pantsing/hector/internal/core"
	"github.com/pantsing/hector/internal/utils"
	"github.com/urfave/cli"
	"os"
	"strconv"
	"strings"
)

func (algo *LogisticRegression) Command() cli.Command {
	return cli.Command{
		Name:     "logRegr",
		Usage:    "Logistic Regression",
		Category: "LR",
		Flags: []cli.Flag{
			cli.Float64Flag{
				Name:  "learning-rate,lrate",
				Value: 0.01,
			},
			cli.Float64Flag{
				Name:  "regularization,r",
				Value: 0.01,
			},
		},
	}
}

type LogisticRegressionParams struct {
	LearningRate   float64
	Regularization float64
	Steps          int
}

type LogisticRegression struct {
	Model  map[int64]float64
	Params LogisticRegressionParams
}

func (algo *LogisticRegression) SaveModel(path string) {
	sb := utils.StringBuilder{}
	for f, g := range algo.Model {
		sb.Int64(f)
		sb.Write("\t")
		sb.Float(g)
		sb.Write("\n")
	}
	sb.WriteToFile(path)
}

func (algo *LogisticRegression) LoadModel(path string) {
	file, _ := os.Open(path)
	defer file.Close()
	algo.Model = make(map[int64]float64)
	scaner := bufio.NewScanner(file)
	for scaner.Scan() {
		line := scaner.Text()
		tks := strings.Split(line, "\t")
		fid, _ := strconv.ParseInt(tks[0], 10, 64)
		fw, _ := strconv.ParseFloat(tks[1], 64)
		algo.Model[fid] = fw
	}
}

func (algo *LogisticRegression) Init(ctx *cli.Context) {
	algo.Model = make(map[int64]float64)
	algo.Params.LearningRate = ctx.Float64("learning-rate")
	algo.Params.Regularization = ctx.Float64("regularization")
	algo.Params.Steps = ctx.Int("steps")
}

func (algo *LogisticRegression) Clear() {
	algo.Model = nil
	algo.Model = make(map[int64]float64)
}

func (algo *LogisticRegression) Train(dataset *core.DataSet) {
	algo.Model = make(map[int64]float64)
	for step := 0; step < algo.Params.Steps; step++ {
		for _, sample := range dataset.Samples {
			prediction := algo.Predict(sample)
			err := sample.LabelDoubleValue() - prediction
			for _, feature := range sample.Features {
				model_feature_value, ok := algo.Model[feature.Id]
				if !ok {
					model_feature_value = 0.0
				}
				model_feature_value += algo.Params.LearningRate * (err*feature.Value - algo.Params.Regularization*model_feature_value)
				algo.Model[feature.Id] = model_feature_value
			}
		}
		algo.Params.LearningRate *= 0.9
	}
}

func (algo *LogisticRegression) Predict(sample *core.Sample) float64 {
	ret := 0.0
	for _, feature := range sample.Features {
		model_feature_value, ok := algo.Model[feature.Id]
		if ok {
			ret += model_feature_value * feature.Value
		}
	}
	return utils.Sigmoid(ret)
}
