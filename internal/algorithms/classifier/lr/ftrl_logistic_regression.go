package lr

import (
	"bufio"
	"github.com/pantsing/hector/internal/core"
	"github.com/pantsing/hector/internal/utils"
	"github.com/qiniu/log"
	"github.com/urfave/cli"
	"math"
	"os"
	"strconv"
	"strings"
)

func (algo *FTRLLogisticRegression) Command() cli.Command {
	return cli.Command{
		Name:  "ftrl",
		Usage: "FTRL Logistic Regeression",
		Flags: []cli.Flag{
			cli.Float64Flag{
				Name:  "alpha,a",
				Value: 0.1,
			},
			cli.Float64Flag{
				Name:  "beta,b",
				Value: 1,
			},
			cli.Float64Flag{
				Name:  "lambda1",
				Value: 0.1,
			},
			cli.Float64Flag{
				Name:  "lambda2",
				Value: 0.1,
			},
			cli.IntFlag{
				Name:  "steps",
				Value: 60,
			},
			cli.BoolTFlag{
				Name: "balance",
			},
			cli.Float64Flag{
				Name:  "subsampleRate,ssr",
				Value: 1,
			},
		},
	}
}

type FTRLLogisticRegressionParams struct {
	Alpha     float64
	Beta      float64
	Lambda1   float64
	Lambda2   float64
	IsBalance bool    // 正反例样本是否均衡
	SSR       float64 // 欠采样比例Sub-Sample Ratio (0,1]。默认值为1,表示正反例样本数量均衡,不均衡时表示负例(label=0)样本。
	SBR       float64 // Samples Balance Ratio 样本集正反例样本比率 (0,1]。默认值为1,表示正反例样本数量均衡。
	Steps     int     // 最大迭代次数
}

type FTRLFeatureWeight struct {
	ni, zi float64
}

func (w *FTRLFeatureWeight) Wi(p FTRLLogisticRegressionParams) float64 {
	wi := 0.0
	if math.Abs(w.zi) > p.Lambda1 {
		wi = (utils.Signum(w.zi)*p.Lambda1 - w.zi) / (p.Lambda2 + (p.Beta+math.Sqrt(w.ni))/p.Alpha)
	}
	return wi
}

type FTRLLogisticRegression struct {
	Model  map[int64]FTRLFeatureWeight
	Params FTRLLogisticRegressionParams
}

func (algo *FTRLLogisticRegression) SaveModel(path string) {
	sb := utils.StringBuilder{}
	for f, g := range algo.Model {
		sb.Int64(f)
		sb.Write("\t")
		sb.Float(g.ni)
		sb.Write("\t")
		sb.Float(g.zi)
		sb.Write("\n")
	}
	sb.WriteToFile(path)
}

func (algo *FTRLLogisticRegression) LoadModel(path string) {
	file, _ := os.Open(path)
	defer file.Close()

	scaner := bufio.NewScanner(file)
	for scaner.Scan() {
		line := scaner.Text()
		tks := strings.Split(line, "\t")
		fid, _ := strconv.ParseInt(tks[0], 10, 64)
		ni, _ := strconv.ParseFloat(tks[1], 64)
		zi, _ := strconv.ParseFloat(tks[2], 64)
		g := FTRLFeatureWeight{ni: ni, zi: zi}
		algo.Model[fid] = g
	}
}

func (algo *FTRLLogisticRegression) Predict(sample *core.Sample) float64 {
	ret := 0.0
	for _, feature := range sample.Features {
		model_feature_value, ok := algo.Model[feature.Id]
		if ok {
			ret += model_feature_value.Wi(algo.Params) * feature.Value
		}
	}
	return utils.Sigmoid(ret)
}

func (algo *FTRLLogisticRegression) Init(ctx *cli.Context) {
	algo.Model = make(map[int64]FTRLFeatureWeight)
	algo.Params.Alpha = ctx.Float64("alpha")
	algo.Params.Beta = ctx.Float64("beta")
	algo.Params.Lambda1 = ctx.Float64("lambda1")
	algo.Params.Lambda2 = ctx.Float64("lambda2")
	algo.Params.Steps = ctx.Int("steps")
	algo.Params.IsBalance = ctx.BoolT("balance")
	algo.Params.SSR = ctx.Float64("subsampleRate")
	if algo.Params.IsBalance {
		algo.Params.SSR = 1
		algo.Params.SBR = 1
	}
	log.Info(algo.Params)
}

func (algo *FTRLLogisticRegression) Clear() {
	algo.Model = nil
	algo.Model = make(map[int64]FTRLFeatureWeight)
}

func (algo *FTRLLogisticRegression) Train(dataset *core.DataSet) {
	n := float64(len(dataset.Samples))
	labelDist := make(map[int]float64, 2)
	for i := range dataset.Samples {
		labelDist[dataset.Samples[i].Label]++
	}
	algo.Params.SBR = labelDist[1] / n

	log.Infof("SBR:%.9g\t SSR:%.9g", algo.Params.SBR, algo.Params.SSR)
	for step := 0; step < algo.Params.Steps; step++ {
		for _, sample := range dataset.Samples {
			prediction := algo.Predict(sample)
			err := sample.LabelDoubleValue() - prediction
			if !algo.Params.IsBalance && sample.Label != 1 {
				err /= algo.Params.SSR
			}
			for _, feature := range sample.Features {
				model_feature_value, ok := algo.Model[feature.Id]
				if !ok {
					model_feature_value = FTRLFeatureWeight{0.0, 0.0}
				}
				zi := model_feature_value.zi
				ni := model_feature_value.ni
				gi := -1 * err * feature.Value
				sigma := (math.Sqrt(ni+gi*gi) - math.Sqrt(ni)) / algo.Params.Alpha
				wi := model_feature_value.Wi(algo.Params)
				zi += gi - sigma*wi
				ni += gi * gi
				algo.Model[feature.Id] = FTRLFeatureWeight{zi: zi, ni: ni}
			}
		}
	}
}
