import json
import os
import tempfile
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


class COCOScoreEvaluator:
    """
    封装 COCO 评测标准 (BLEU, METEOR, ROUGE-L, CIDEr, SPICE)
    """

    def __init__(self):
        pass

    def evaluate(self, ground_truth, predictions, fast_mode=False):
        """
        计算评测指标

        参数:
            ground_truth: dict, 格式 {image_id: ["caption 1", "caption 2", ...]}
            predictions: dict, 格式 {image_id: ["generated caption"]}
            fast_mode: bool, 是否启用快速模式 (只计算 CIDEr 和 BLEU)

        返回:
            scores: dict, 各项指标的分数
        """
        print("正在构建 COCO 格式数据...")

        # 1. 创建临时文件用于存放 COCO 格式的 JSON
        # 因为 pycocoevalcap 需要读取文件
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f_gt:
            coco_gt_path = f_gt.name
            json.dump(self._format_to_coco_gt(ground_truth), f_gt)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f_res:
            coco_res_path = f_res.name
            json.dump(self._format_to_coco_res(predictions), f_res)

        try:
            # 2. 初始化 COCO 对象
            coco = COCO(coco_gt_path)
            coco_result = coco.loadRes(coco_res_path)

            # 3. 初始化评测对象
            coco_eval = COCOEvalCap(coco, coco_result)
            coco_eval.params["image_id"] = coco_result.getImgIds()

            # === 修改开始 ===
            if fast_mode:
                # 训练时只计算 CIDEr 和 BLEU
                print("快速评测模式: 只计算 CIDEr 和 BLEU...")

                # 关键修复：不手动构建 gts/res，而是利用 tokenizer 预处理
                # 1. 先运行分词 (这是必须的步骤)
                from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

                tokenizer = PTBTokenizer()
                coco_eval.imgToAnns = {
                    imgId: coco_eval.coco.imgToAnns[imgId]
                    for imgId in coco_eval.params["image_id"]
                }
                coco_eval.gts = tokenizer.tokenize(coco_eval.imgToAnns)
                coco_eval.res = tokenizer.tokenize(coco_eval.cocoRes.imgToAnns)

                # 2. 手动调用需要的 scorer
                from pycocoevalcap.bleu.bleu import Bleu
                from pycocoevalcap.cider.cider import Cider

                scorers = [
                    (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                    (Cider(), "CIDEr"),
                ]

                eval_res = {}
                for scorer, method in scorers:
                    # 现在 gts 和 res 已经是分词后的格式了，可以直接计算
                    score, scores = scorer.compute_score(coco_eval.gts, coco_eval.res)
                    if isinstance(method, list):
                        for sc, scs, m in zip(score, scores, method):
                            eval_res[m] = sc
                    else:
                        eval_res[method] = score

                coco_eval.eval = eval_res
            else:
                # === 修改结束 ===
                # 4. 执行完整评测
                print("开始计算所有指标...")
                coco_eval.evaluate()
            # ===============

            # 5. 提取结果
            scores = coco_eval.eval
            return scores

        except Exception as e:
            print(f"评测过程中出错: {e}")
            return {}
        finally:
            # 清理临时文件
            if os.path.exists(coco_gt_path):
                os.remove(coco_gt_path)
            if os.path.exists(coco_res_path):
                os.remove(coco_res_path)

    def _format_to_coco_gt(self, ground_truth):
        """将数据转换为 COCO Ground Truth 格式"""
        images = []
        annotations = []
        ann_id = 1

        for img_id, captions in ground_truth.items():
            images.append({"id": img_id})
            for cap in captions:
                annotations.append({"image_id": img_id, "id": ann_id, "caption": cap})
                ann_id += 1

        return {
            "images": images,
            "annotations": annotations,
            "type": "captions",
            "info": "dummy",
            "licenses": "dummy",
        }

    def _format_to_coco_res(self, predictions):
        """将数据转换为 COCO Results 格式"""
        results = []
        for img_id, caps in predictions.items():
            # 通常预测只有一个 caption
            cap = caps[0] if isinstance(caps, list) else caps
            results.append({"image_id": img_id, "caption": cap})
        return results


if __name__ == "__main__":
    # 测试代码
    gt = {1: ["a red dress", "a beautiful red dress"], 2: ["blue jeans"]}
    pred = {1: ["red dress"], 2: ["blue pants"]}

    evaluator = COCOScoreEvaluator()
    scores = evaluator.evaluate(gt, pred, fast_mode=True)
    print("测试分数:", scores)
