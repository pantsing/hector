package eval

import (
	"github.com/qiniu/log"
	"math"
	"sort"
)

type LabelPrediction struct {
	Prediction float64
	Label      int
}

type RealPrediction struct { // Real valued
	Prediction float64
	Value      float64
}

type By func(p1, p2 *LabelPrediction) bool

type labelPredictionSorter struct {
	predictions []*LabelPrediction
	by          By
}

func (s *labelPredictionSorter) Len() int {
	return len(s.predictions)
}

func (s *labelPredictionSorter) Swap(i, j int) {
	s.predictions[i], s.predictions[j] = s.predictions[j], s.predictions[i]
}

func (s *labelPredictionSorter) Less(i, j int) bool {
	return s.by(s.predictions[i], s.predictions[j])
}

func (by By) Sort(predictions []*LabelPrediction) {
	sorter := &labelPredictionSorter{
		predictions: predictions,
		by:          by,
	}
	sort.Sort(sorter)
}

func AUC(predictions0 []*LabelPrediction) float64 {
	predictions := []*LabelPrediction{}
	for _, pred := range predictions0 {
		predictions = append(predictions, pred)
	}
	prediction := func(p1, p2 *LabelPrediction) bool {
		return p1.Prediction > p2.Prediction
	}

	By(prediction).Sort(predictions)

	pn := 0.0
	nn := float64(len(predictions))
	ret := 0.0
	count := nn
	for i, lp := range predictions {
		if lp.Label > 0 {
			pn += 1.0
			nn -= 1.0
			ret += float64(count) - float64(i)
		}
	}
	ret2 := pn * (pn + 1) / 2.0
	if pn*nn == 0.0 {
		return 0.5
	}
	return (ret - ret2) / (pn * nn)
}

func RMSE(predictions []*LabelPrediction) float64 {
	ret := 0.0
	n := 0.0

	for _, pred := range predictions {
		ret += (float64(pred.Label) - pred.Prediction) * (float64(pred.Label) - pred.Prediction)
		n += 1.0
	}

	return math.Sqrt(ret / n)
}

func ErrorRate(predictions []*LabelPrediction) float64 {
	var FN, FP, TN, TP float64
	ret := 0.0
	n := float64(len(predictions))

	for _, pred := range predictions {
		if float64(pred.Label) >= 0.5 {
			if pred.Prediction < 0.5 {
				ret++
				FN++
			} else {
				TP++
			}
		} else {
			if pred.Prediction >= 0.5 {
				ret++
				FP++
			} else {
				TN++
			}
		}
	}
	log.Infof("FN: %.9g\tFP: %.9g\tTN: %.9g\tTP: %.9g\tPrecision: %.9g\tRecall: %.9g\t", FN, FP, TN, TP, TP/(TP+FP), TP/(TP+FN))
	return ret / n
}

// BalanceErrorRate considers the real ratio between positive and negative samples.
// Sample labels should be marked 1 for positive samples and 0 for negative.
// The ssr is sub-sample rate (0,1]. If no sub-sample applied on sample set, ssr should be set as 1.
// Sub-sample is on negative samples.
func BalanceErrorRate(predictions []*LabelPrediction, ssr float64) float64 {
	ret := 0.0
	n := float64(len(predictions))

	labelDist := make(map[int]float64, 2)
	for i := range predictions {
		labelDist[predictions[i].Label]++
	}
	r := ssr * (labelDist[1] / n)

	for _, pred := range predictions {
		p := pred.Prediction / (pred.Prediction + (1-pred.Prediction)/r)
		if (float64(pred.Label)-0.5)*(p-0.5) < 0 {
			ret += 1.0
		}
	}
	return ret / n
}

func RegRMSE(predictions []*RealPrediction) float64 {
	ret := 0.0
	n := 0.0

	for _, pred := range predictions {
		ret += (pred.Value - pred.Prediction) * (pred.Value - pred.Prediction)
		n += 1.0
	}

	return math.Sqrt(ret / n)
}
