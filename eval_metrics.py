import json
import os
import tempfile
import subprocess
import threading
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.meteor.meteor import Meteor

class FixedMeteor(Meteor):
    """
    修复版 Meteor 类，解决了 Java 命令参数顺序错误导致的 Broken pipe 问题。
    """
    def __init__(self):
        # 获取原始 meteor 模块的路径以找到 jar 文件
        import pycocoevalcap.meteor.meteor as original_meteor_module
        meteor_dir = os.path.dirname(os.path.abspath(original_meteor_module.__file__))
        meteor_jar = 'meteor-1.5.jar'
        
        # 修复: 将 -Xmx2G 放在 -jar 之前
        self.meteor_cmd = ['java', '-Xmx2G', '-jar', meteor_jar, \
                '-', '-', '-stdio', '-l', 'en', '-norm']
        self.meteor_p = subprocess.Popen(self.meteor_cmd, \
                cwd=meteor_dir, \
                stdin=subprocess.PIPE, \
                stdout=subprocess.PIPE, \
                stderr=subprocess.PIPE)
        self.lock = threading.Lock()

class COCOScoreEvaluator:
    """
    封装 COCO 评测标准 (METEOR, ROUGE-L, CIDEr, SPICE)
    """

    def __init__(self):
        pass

    def evaluate(self, ground_truth, predictions, fast_mode=False):
        """
        计算评测指标

        参数:
            ground_truth: dict, 格式 {image_id: ["caption 1", "caption 2", ...]}
            predictions: dict, 格式 {image_id: ["generated caption"]}
            fast_mode: bool, 是否启用快速模式 (只计算 CIDEr 和 METEOR，跳过 SPICE)

        返回:
            scores: dict, 各项指标的分数
        """
        print("正在构建 COCO 格式数据...")

        # 1. 创建临时文件用于存放 COCO 格式的 JSON
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

            if fast_mode:
                # 快速模式: 只计算 CIDEr, METEOR, ROUGE-L (跳过耗时的 SPICE)
                print("快速评测模式: 计算 CIDEr, METEOR, ROUGE-L (跳过 SPICE)...")

                from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

                tokenizer = PTBTokenizer()
                coco_eval.imgToAnns = {
                    imgId: coco_eval.coco.imgToAnns[imgId]
                    for imgId in coco_eval.params["image_id"]
                }
                coco_eval.gts = tokenizer.tokenize(coco_eval.imgToAnns)
                coco_eval.res = tokenizer.tokenize(coco_eval.cocoRes.imgToAnns)

                # 手动调用需要的 scorer (不包括 SPICE，因为太慢)
                from pycocoevalcap.cider.cider import Cider
                from pycocoevalcap.rouge.rouge import Rouge

                scorers = [
                    (Cider(), "CIDEr"),
                    (FixedMeteor(), "METEOR"),
                    (Rouge(), "ROUGE_L"),
                ]

                eval_res = {}
                for scorer, method in scorers:
                    try:
                        score, scores = scorer.compute_score(
                            coco_eval.gts, coco_eval.res
                        )
                        eval_res[method] = score
                        print(f"  {method}: {score:.4f}")
                    except Exception as e:
                        print(f"  {method} 计算失败: {e}")
                        eval_res[method] = 0.0

                # 快速模式下 SPICE 设为 0
                eval_res["SPICE"] = 0.0
                print("  SPICE: 跳过 (快速模式)")

                coco_eval.eval = eval_res
            else:
                # 完整模式: 计算所有四个指标 (METEOR, ROUGE-L, CIDEr, SPICE)
                print("完整评测模式: 计算 METEOR, ROUGE-L, CIDEr, SPICE...")

                from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

                tokenizer = PTBTokenizer()
                coco_eval.imgToAnns = {
                    imgId: coco_eval.coco.imgToAnns[imgId]
                    for imgId in coco_eval.params["image_id"]
                }
                coco_eval.gts = tokenizer.tokenize(coco_eval.imgToAnns)
                coco_eval.res = tokenizer.tokenize(coco_eval.cocoRes.imgToAnns)

                from pycocoevalcap.cider.cider import Cider
                from pycocoevalcap.rouge.rouge import Rouge
                from pycocoevalcap.spice.spice import Spice

                scorers = [
                    (Cider(), "CIDEr"),
                    (FixedMeteor(), "METEOR"),
                    (Rouge(), "ROUGE_L"),
                    (Spice(), "SPICE"),
                ]

                eval_res = {}
                for scorer, method in scorers:
                    try:
                        print(f"  正在计算 {method}...")
                        score, scores = scorer.compute_score(
                            coco_eval.gts, coco_eval.res
                        )
                        eval_res[method] = score
                        print(f"  {method}: {score:.4f}")
                    except Exception as e:
                        print(f"  {method} 计算失败: {e}")
                        eval_res[method] = 0.0

                coco_eval.eval = eval_res

            # 5. 提取结果
            scores = coco_eval.eval
            return scores

        except Exception as e:
            print(f"评测过程中出错: {e}")
            import traceback

            traceback.print_exc()
            return {"CIDEr": 0.0, "METEOR": 0.0, "ROUGE_L": 0.0, "SPICE": 0.0}
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
            cap = caps[0] if isinstance(caps, list) else caps
            results.append({"image_id": img_id, "caption": cap})
        return results


if __name__ == "__main__":
    # 测试代码
    gt = {1: ["a red dress", "a beautiful red dress"], 2: ["blue jeans"]}
    pred = {1: ["red dress"], 2: ["blue pants"]}

    evaluator = COCOScoreEvaluator()
    print("测试快速模式:")
    scores = evaluator.evaluate(gt, pred, fast_mode=True)
    print("测试分数:", scores)

    print("\n测试完整模式 (包含 SPICE):")
    scores_full = evaluator.evaluate(gt, pred, fast_mode=False)
    print("完整模式分数:", scores_full)
