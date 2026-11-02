"""意图分类单元测试：覆盖评测驱动的关键回归用例。"""
from embroidery_agent.nlp.intent import IntentClassifier


def test_customer_knowledge():
    clf = IntentClassifier()
    assert clf.classify("苏绣有什么特点", "customer").intent.value == "knowledge"
    assert clf.classify("打籽绣是什么针法", "customer").intent.value == "knowledge"
    assert clf.classify("什么是定制肖像绣", "customer").intent.value == "knowledge"
    assert clf.classify("湘绣最擅长绣什么", "customer").intent.value == "knowledge"
    assert clf.classify("装裱刺绣有什么讲究", "customer").intent.value == "knowledge"
    assert clf.classify("中式家装适合挂什么绣品", "customer").intent.value == "knowledge"


def test_customer_pattern_meaning():
    clf = IntentClassifier()
    assert clf.classify("牡丹纹样有什么寓意", "customer").intent.value == "pattern_meaning"
    assert clf.classify("送长辈选什么纹样好", "customer").intent.value == "pattern_meaning"
    assert clf.classify("婚庆送刺绣选什么题材", "customer").intent.value == "pattern_meaning"
    assert clf.classify("年年有余是什么纹样", "customer").intent.value == "pattern_meaning"
    assert clf.classify("送老师绣什么题材", "customer").intent.value == "pattern_meaning"


def test_customer_custom_requirement():
    clf = IntentClassifier()
    assert clf.classify("我想定制一幅牡丹挂画", "customer").intent.value == "custom_requirement"
    assert clf.classify("想绣一幅山水，120乘60的", "customer").intent.value == "custom_requirement"
    assert clf.classify("帮我绣一张照片做成挂画", "customer").intent.value == "custom_requirement"
    assert clf.classify("能做一幅刺绣旗袍吗，绣兰花", "customer").intent.value == "custom_requirement"


def test_customer_order_price_transfer():
    clf = IntentClassifier()
    assert clf.classify("一幅绣品多少钱", "customer").intent.value == "order_service"
    assert clf.classify("现在买有折扣吗", "customer").intent.value == "order_service"
    assert clf.classify("定制一幅要多久", "customer").intent.value == "order_service"
    assert clf.classify("有现货吗", "customer").intent.value == "order_service"
    assert clf.classify("绣品可以退货吗", "customer").intent.value == "order_service"
    assert clf.classify("能不能便宜点", "customer").intent.value == "order_service"


def test_customer_complaint():
    clf = IntentClassifier()
    assert clf.classify("你们的绣品质量太差了", "customer").intent.value == "complaint"
    assert clf.classify("这个绣品有色差，和图片不符", "customer").intent.value == "complaint"


def test_customer_custom_weak_signals():
    """不带"绣/定制"强词的口语问法也应触发定制（修复哑火回归）。"""
    clf = IntentClassifier()
    assert clf.classify("能做清明上河图吗", "customer").intent.value == "custom_requirement"
    assert clf.classify("我想买个清明上河图", "customer").intent.value == "custom_requirement"
    assert clf.classify("绣朵牡丹送长辈好吗", "customer").intent.value == "custom_requirement"
    assert clf.classify("你们能定做清明上河图吗", "customer").intent.value == "custom_requirement"
    assert clf.classify("想买一幅双面绣送人", "customer").intent.value == "custom_requirement"


def test_customer_pattern_vs_custom():
    """寓意问优先于弱定制词；强定制词优先于寓意问。"""
    clf = IntentClassifier()
    assert clf.classify("牡丹送长辈有什么寓意", "customer").intent.value == "pattern_meaning"
    assert clf.classify("莲花象征什么", "customer").intent.value == "pattern_meaning"
    assert clf.classify("定制牡丹挂画有什么寓意", "customer").intent.value == "custom_requirement"


def test_customer_custom_neg():
    """"哪些/种类"等知识问法不得误判为定制意向。"""
    clf = IntentClassifier()
    assert clf.classify("你们能做哪些绣品", "customer").intent.value == "knowledge"
    assert clf.classify("刺绣有哪些种类", "customer").intent.value == "knowledge"
    assert clf.classify("宝宝满月送什么绣品", "customer").intent.value == "knowledge"
    # 咨询性表述："适合送人/外国风格"是知识，不是定制意向
    assert clf.classify("刺绣围巾披肩适合送人吗", "customer").intent.value == "knowledge"
    assert clf.classify("能绣外国油画风格的作品吗", "customer").intent.value == "knowledge"


def test_staff_knowledge_not_order():
    """员工场景：订单/价格/投诉信号词应视为内部知识，而非客户订单。"""
    clf = IntentClassifier()
    assert clf.classify("手工绣的工期标准是什么", "staff").intent.value == "knowledge"
    assert clf.classify("运费按什么规则核算", "staff").intent.value == "knowledge"
    assert clf.classify("装裱成本怎么核算", "staff").intent.value == "knowledge"
    assert clf.classify("批量订单怎么报价", "staff").intent.value == "knowledge"
    assert clf.classify("客户投诉物流慢怎么处理", "staff").intent.value == "knowledge"
    assert clf.classify("客户压价时内部怎么处理", "staff").intent.value == "knowledge"


def test_staff_tools():
    clf = IntentClassifier()
    assert clf.classify("复盘一下这段客户对话", "staff").intent.value == "staff_analysis"
    assert clf.classify("写一条刺绣挂画的朋友圈文案", "staff").intent.value == "staff_copywriting"
