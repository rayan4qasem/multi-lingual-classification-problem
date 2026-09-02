"""Synthetic Arabic document generation.

Two engines, because they answer different questions.

`template` is free, offline and deterministic under a seed. Use it to
exercise the plumbing — ingestion, batching, evaluation, the CLI — without
spending a riyal. Its weakness is honest: the bodies are drawn from
per-institution pools, so a keyword model scores unrealistically well on it.

`llm` asks Claude to write documents that read like real correspondence,
including deliberately ambiguous ones that sit on a confusion boundary. Use
it for any accuracy number you intend to quote.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import anthropic
import yaml

from . import DEFAULT_MODEL
from .models import Document
from .taxonomy import Taxonomy
from .taxonomy import load as load_taxonomy

FIRST_NAMES = [
    "محمد",
    "عبدالله",
    "فهد",
    "سعود",
    "خالد",
    "نورة",
    "سارة",
    "منيرة",
    "عبدالرحمن",
    "تركي",
    "ريم",
    "هند",
    "بندر",
    "ماجد",
    "لطيفة",
    "أحمد",
]
FAMILY_NAMES = [
    "العتيبي",
    "القحطاني",
    "الغامدي",
    "الشهري",
    "الدوسري",
    "الحربي",
    "المطيري",
    "الزهراني",
    "السبيعي",
    "البقمي",
    "العنزي",
    "الشمري",
]
CITIES = [
    "الرياض",
    "جدة",
    "الدمام",
    "مكة المكرمة",
    "المدينة المنورة",
    "أبها",
    "تبوك",
    "بريدة",
    "حائل",
    "الخبر",
    "نجران",
    "جازان",
]
HIJRI_MONTHS = [
    "محرم",
    "صفر",
    "ربيع الأول",
    "ربيع الآخر",
    "جمادى الأولى",
    "جمادى الآخرة",
    "رجب",
    "شعبان",
    "رمضان",
    "شوال",
    "ذو القعدة",
    "ذو الحجة",
]

# Scenario bodies per institution: (subject, body). Each body is written as a
# citizen or an office would write it, not as a keyword list.
SCENARIOS: dict[str, list[tuple[str, str]]] = {
    "interior_public_security": [
        (
            "بلاغ عن سرقة مركبة",
            "أفيد سعادتكم بأنني فوجئت صباح اليوم بفقدان مركبتي من نوع {car} موديل {year} "
            "والمتوقفة أمام مسكني بحي {district}. وقد راجعت مركز الشرطة وقدمت البيانات "
            "كاملة، وأرفق صورة من استمارة المركبة. آمل التكرم بالتوجيه بالبحث عنها.",
        ),
        (
            "شكوى من إزعاج متكرر",
            "أشكو من تجمعات ليلية متكررة أمام المنزل في حي {district} يصاحبها تفحيط "
            "وأصوات مرتفعة حتى ساعات متأخرة، وقد تكرر الأمر أكثر من عشر مرات هذا الشهر "
            "دون أن يتوقف. أرجو تكثيف الدوريات في الموقع.",
        ),
        (
            "طلب صورة من تقرير حادث",
            "تعرضت لحادث مروري بتاريخ {date} على طريق {district}، وقد باشرته الدورية "
            "وحُرر تقرير بالواقعة. أطلب تزويدي بصورة معتمدة من التقرير لاستكمال "
            "إجراءات المطالبة.",
        ),
    ],
    "public_prosecution": [
        (
            "طلب الاطلاع على ملف التحقيق",
            "بصفتي وكيلاً عن المتهم في القضية رقم {case} المحالة إليكم، أطلب التصريح لي "
            "بالاطلاع على أوراق التحقيق وتصوير ما يلزم منها تمكيناً لحق الدفاع، علماً "
            "بأن الاستجواب تم بتاريخ {date}.",
        ),
        (
            "تظلم من أمر توقيف",
            "أتقدم بالتظلم من أمر التوقيف الصادر بحق موكلي في القضية رقم {case}، "
            "لانتفاء مبررات استمرار التوقيف وثبوت محل إقامته الدائم، وأطلب الإفراج "
            "عنه بكفالة حضورية.",
        ),
        (
            "إشعار بإحالة قضية",
            "إشارة إلى محضر الضبط الوارد من الجهة الأمنية بشأن الواقعة المقيدة برقم "
            "{case}، تمت مباشرة التحقيق وسماع أقوال الأطراف، وقد تقرر توجيه الاتهام "
            "وإحالة الأوراق إلى المحكمة المختصة.",
        ),
    ],
    "moj_courts": [
        (
            "صحيفة دعوى مطالبة مالية",
            "أقيم دعواي هذه ضد المدعى عليه المقيم بمدينة {city}، لمطالبته بمبلغ وقدره "
            "{amount} ريال، وهي قيمة سند لأمر حل أجله بتاريخ {date} ولم يسدده رغم "
            "المطالبة. وأطلب إلزامه بالسداد وبالتعويض عن التأخير.",
        ),
        (
            "طلب تنفيذ حكم",
            "صدر لصالحي الحكم رقم {case} والقاضي بإلزام المحكوم عليه بتسليم العقار "
            "الموضح بالصك، وقد اكتسب الحكم القطعية ولم ينفذ طوعاً. أطلب فتح ملف تنفيذ "
            "واتخاذ الإجراءات النظامية.",
        ),
        (
            "طلب إصدار وكالة شرعية",
            "أرغب في توكيل ابني الموضحة بياناته أدناه للقيام مقامي في إدارة أملاكي "
            "بمدينة {city} والمرافعة والمدافعة أمام الجهات المختصة، وذلك لظروفي "
            "الصحية الموضحة في التقرير المرفق.",
        ),
    ],
    "moh_health": [
        (
            "طلب تحويل لمستشفى تخصصي",
            "المريض المذكور أعلاه يعاني من حالة تستدعي متابعة تخصصية غير متوفرة في "
            "المركز، وقد أُجريت له الفحوصات الأولية بتاريخ {date}. يوصى بتحويله إلى "
            "المستشفى التخصصي بمدينة {city} لاستكمال الخطة العلاجية.",
        ),
        (
            "شكوى تأخر موعد",
            "حجزت موعداً في عيادة الباطنة منذ أكثر من أربعة أشهر، وكلما راجعت أُبلغت "
            "بتأجيله دون سبب واضح، رغم أن حالتي تستدعي المتابعة الدورية وصرف الدواء. "
            "أرجو النظر في تقديم الموعد.",
        ),
        (
            "طلب تقرير طبي معتمد",
            "أطلب تزويدي بتقرير طبي معتمد يوضح مدة التنويم والحالة الصحية خلال فترة "
            "علاجي في المستشفى بتاريخ {date}، وذلك لتقديمه لجهة العمل.",
        ),
    ],
    "civil_defense": [
        (
            "بلاغ عن مخالفة اشتراطات السلامة",
            "أفيدكم بوجود مستودع في حي {district} يخزن مواد قابلة للاشتعال دون توفر "
            "طفايات صالحة أو مخارج طوارئ، ويقع بمحاذاة مبنى سكني مأهول. أرجو الكشف "
            "على الموقع واتخاذ اللازم قبل وقوع كارثة.",
        ),
        (
            "طلب تصريح سلامة لمنشأة",
            "بصدد افتتاح منشأة تجارية بمدينة {city}، وقد استكملنا تركيب أنظمة الإنذار "
            "والإطفاء ومخارج الطوارئ وفق المخططات المرفقة. نطلب تحديد موعد للكشف "
            "وإصدار شهادة السلامة.",
        ),
        (
            "تقرير عن حريق",
            "باشر الفريق بلاغاً عن حريق شب في الدور الأرضي بحي {district} الساعة "
            "الثانية فجراً، وتمت السيطرة عليه وإخلاء ساكني المبنى دون إصابات تذكر، "
            "ويرجح أن السبب تماس كهربائي.",
        ),
    ],
    "moe_education": [
        (
            "طلب معادلة شهادة",
            "حصلت على درجة البكالوريوس من جامعة خارج المملكة بتاريخ {date}، وأرفق "
            "الوثائق المصدقة وكشف الدرجات. أطلب معادلة الشهادة لاستكمال إجراءات "
            "التوظيف.",
        ),
        (
            "شكوى ولي أمر بشأن النقل المدرسي",
            "ابني طالب في الصف الرابع بمدرسة حي {district}، وحافلة النقل تتأخر يومياً "
            "أكثر من نصف ساعة عن موعد الحضور، مما أثر على انتظامه. أرجو معالجة الوضع.",
        ),
        (
            "طلب نقل طالب",
            "نظراً لانتقال سكننا من مدينة {city} إلى حي {district}، أطلب نقل ابنتي إلى "
            "أقرب مدرسة لمقر السكن الجديد اعتباراً من الفصل الدراسي القادم.",
        ),
    ],
    "hrsd_labor": [
        (
            "شكوى تأخر صرف الأجور",
            "أعمل لدى المنشأة الموضحة أدناه منذ ثلاث سنوات، وقد تأخر صرف راتبي عن "
            "أربعة أشهر متتالية رغم مراجعتي المتكررة للإدارة. أطلب إلزام صاحب العمل "
            "بصرف مستحقاتي المتأخرة البالغة {amount} ريال.",
        ),
        (
            "طلب نقل خدمات",
            "أتقدم بطلب نقل خدماتي إلى منشأة أخرى، لعدم التزام صاحب العمل الحالي "
            "ببنود العقد الموقع بتاريخ {date} فيما يخص ساعات العمل والإجازة السنوية.",
        ),
        (
            "اعتراض على إنهاء عقد",
            "فوجئت بإنهاء عقدي دون إشعار مسبق ودون مبرر نظامي بعد خدمة تجاوزت خمس "
            "سنوات. أطلب النظر في التعويض عن الفصل ومستحقات نهاية الخدمة.",
        ),
    ],
    "gosi": [
        (
            "طلب صرف معاش تقاعدي",
            "أكملت مدة الاشتراك النظامية وبلغت السن المقررة، وأتقدم بطلب صرف المعاش "
            "التقاعدي اعتباراً من تاريخ {date}، مرفقاً به بيان مدة الاشتراك وصورة "
            "الهوية والحساب البنكي.",
        ),
        (
            "اعتراض على مدة اشتراك",
            "لاحظت عند استخراج بيان الاشتراك وجود فترة عمل من عام {year} غير مسجلة "
            "رغم أنني كنت على رأس العمل خلالها. أطلب تصحيح البيان وضم المدة.",
        ),
        (
            "إشعار إصابة عمل",
            "تعرض المشترك الموضح أعلاه لإصابة أثناء أداء عمله بتاريخ {date} نتج عنها "
            "عجز جزئي، وأرفق التقرير الطبي وتقرير المنشأة. يرجى تقدير نسبة العجز "
            "وصرف التعويض المستحق.",
        ),
    ],
    "zatca": [
        (
            "اعتراض على ربط ضريبي",
            "استلمت إشعار الربط رقم {case} المتضمن فروقات في ضريبة القيمة المضافة عن "
            "الفترة المنتهية في {date}. لدينا مستندات تثبت صحة الإقرار المقدم، ونتقدم "
            "بالاعتراض خلال المدة النظامية.",
        ),
        (
            "استفسار عن الفوترة الإلكترونية",
            "منشأتنا مسجلة في ضريبة القيمة المضافة، ونستفسر عن المتطلبات الفنية "
            "لربط نظام الفوترة لدينا بالمنصة، وعن المهلة المحددة لشريحتنا.",
        ),
        (
            "طلب فسح إرسالية",
            "لدينا إرسالية واردة عبر منفذ {city} بموجب البيان الجمركي رقم {case}، "
            "وقد استُكملت المستندات وشهادة المنشأ. نطلب سرعة إنهاء إجراءات الفسح.",
        ),
    ],
    "mci_commerce": [
        (
            "بلاغ عن غش تجاري",
            "اشتريت جهازاً من محل بحي {district} على أنه أصلي بضمان الوكيل، وتبين "
            "لاحقاً أنه مقلد ولا يقبله مركز الصيانة المعتمد. أرفق الفاتورة وأطلب "
            "اتخاذ الإجراء النظامي بحق المحل.",
        ),
        (
            "طلب تعديل سجل تجاري",
            "أرغب في تعديل النشاط المدون في السجل التجاري رقم {case} وإضافة نشاط "
            "جديد، مع تحديث عنوان المقر إلى مدينة {city}.",
        ),
        (
            "شكوى امتناع عن تنفيذ ضمان",
            "تعطل المنتج خلال فترة الضمان الموضحة في الفاتورة المؤرخة في {date}، "
            "ويرفض التاجر الإصلاح أو الاستبدال بحجة سوء الاستخدام دون تقرير فني.",
        ),
    ],
    "momah_municipal": [
        (
            "طلب رخصة بناء",
            "أملك أرضاً بحي {district} بموجب الصك المرفق، وأرغب في استخراج رخصة بناء "
            "لعمارة سكنية وفق المخططات المعتمدة من المكتب الهندسي المرفقة.",
        ),
        (
            "بلاغ عن تراكم نفايات",
            "تتراكم النفايات في الشارع المجاور لمنزلي بحي {district} منذ أسبوعين دون "
            "رفع، وقد تسببت في روائح وانتشار حشرات. أرجو التوجيه بالمعالجة.",
        ),
        (
            "شكوى إشغال رصيف",
            "قام محل تجاري في حي {district} بوضع بضائع ومظلات تشغل الرصيف بالكامل "
            "وتجبر المشاة على السير في الشارع، مما يشكل خطراً على السلامة.",
        ),
    ],
    "mot_transport": [
        (
            "شكوى عن حالة طريق",
            "الطريق الرابط بين {city} والمحافظة المجاورة يعاني من حفر عميقة وغياب "
            "الإنارة على امتداد عدة كيلومترات، وقد تسبب في حوادث متكررة. أرجو "
            "إدراجه ضمن خطط الصيانة.",
        ),
        (
            "طلب ترخيص نشاط نقل",
            "أرغب في الحصول على ترخيص لمزاولة نشاط نقل البضائع بالشاحنات، وأرفق "
            "السجل التجاري وبيانات الأسطول المكون من {count} شاحنات.",
        ),
        (
            "شكوى ضد شركة نقل ركاب",
            "حجزت رحلة على حافلة من {city} بتاريخ {date}، وتأخر الانطلاق أربع ساعات "
            "دون إشعار أو تعويض، ورفضت الشركة رد قيمة التذكرة.",
        ),
    ],
    "civil_affairs": [
        (
            "طلب إصدار بدل فاقد للهوية",
            "فقدت هويتي الوطنية أثناء سفري بتاريخ {date}، وقد أبلغت الجهة المختصة. "
            "أتقدم بطلب إصدار بدل فاقد، مرفقاً به صورة من سجل الأسرة.",
        ),
        (
            "طلب تسجيل مولود",
            "رزقت بمولود بتاريخ {date} في مستشفى بمدينة {city}، وأرفق إشعار الولادة. "
            "أطلب تسجيله في سجل الأسرة واستخراج شهادة الميلاد.",
        ),
        (
            "طلب تصحيح اسم في السجل",
            "ورد اسم والدتي في سجل الأسرة مخالفاً لما هو مثبت في وثائقها الرسمية، "
            "وأطلب تصحيح القيد وفق المستندات المرفقة.",
        ),
    ],
    "mewa_environment": [
        (
            "شكوى انقطاع المياه",
            "ينقطع التيار المائي عن حي {district} بشكل شبه يومي لفترات طويلة، "
            "ونضطر لشراء الوايتات على حسابنا الخاص. أرجو معالجة ضعف الضخ في الحي.",
        ),
        (
            "بلاغ عن تلوث",
            "تقوم إحدى المنشآت قرب مدينة {city} بتصريف مخلفات سائلة في مجرى الوادي، "
            "مما أدى إلى تغير لون المياه ونفوق أسماك. أرجو الكشف الميداني.",
        ),
        (
            "طلب رخصة مزرعة",
            "أرغب في استخراج رخصة لمزرعة إنتاج نباتي على أرض مساحتها {count} هكتار "
            "بمنطقة {city}، مع طلب تصريح حفر بئر وفق الاشتراطات.",
        ),
    ],
}

# Documents written to sit on a confusion boundary. The label is the correct
# answer; the text deliberately carries surface signals of the other class.
HARD_CASES: list[tuple[str, str, str]] = [
    (
        "moj_courts",
        "مطالبة بأجور بعد صدور حكم",
        "صدر لصالحي حكم من الدائرة العمالية بإلزام المنشأة بدفع أجوري المتأخرة "
        "البالغة {amount} ريال، وقد اكتسب القطعية ولم تنفذه المنشأة. أطلب فتح ملف "
        "تنفيذ ضدها. علماً بأن الشكوى العمالية سبق أن قُدمت وانتهت بالحكم المذكور.",
    ),
    (
        "hrsd_labor",
        "إصابة أثناء العمل ومطالبة بالأجر",
        "أصبت أثناء عملي بتاريخ {date} ونُقلت للمستشفى، ورفض صاحب العمل صرف أجري "
        "عن فترة الانقطاع رغم أن الإصابة وقعت في موقع العمل. مطالبتي هنا بالأجر "
        "المحتجز وليس بتعويض العجز.",
    ),
    (
        "civil_defense",
        "حريق في مبنى تحت الإنشاء",
        "شب حريق في مبنى تحت الإنشاء بحي {district} يحمل رخصة بناء سارية. تمت "
        "السيطرة عليه، ويتبين من الكشف عدم توفر اشتراطات السلامة في موقع العمل، "
        "ويلزم إيقاف العمل حتى استيفائها.",
    ),
    (
        "mci_commerce",
        "فاتورة غير نظامية من محل",
        "طلبت من المحل فاتورة ضريبية عن مشترياتي فرفض إصدارها وأعطاني ورقة غير "
        "رسمية، ثم تبين أن السلعة مقلدة أصلاً وليست من الوكيل المعتمد. شكواي على "
        "المحل وممارساته التجارية.",
    ),
    (
        "interior_public_security",
        "بلاغ عن اعتداء قبل التحقيق",
        "تعرضت لاعتداء بالضرب من شخص مجهول بحي {district} بتاريخ {date}، وأرغب في "
        "تقديم بلاغ رسمي وضبط الجاني. لم يُباشر التحقيق بعد ولم تُحل القضية لأي جهة.",
    ),
    (
        "momah_municipal",
        "مطعم يصرف مخلفاته في الشارع",
        "يقوم مطعم في حي {district} بتصريف مياه ومخلفات المطبخ إلى الشارع العام، "
        "مما أدى إلى تجمعات مائية وروائح. المطلوب مخالفة المحل وإلزامه بالتصريف "
        "النظامي.",
    ),
]

OPENINGS = [
    "سعادة مدير عام {org} المحترم\nالسلام عليكم ورحمة الله وبركاته،",
    "المكرم/ مدير {org}\t\t\tحفظه الله\nالسلام عليكم ورحمة الله وبركاته، وبعد:",
    "إلى: {org}\nالموضوع: {subject}\nتحية طيبة وبعد،",
]
CLOSINGS = [
    "وتفضلوا بقبول خالص التحية والتقدير،",
    "شاكراً لكم حسن تعاونكم، ولكم جزيل الشكر.",
    "آمل التكرم بالاطلاع واتخاذ ما ترونه مناسباً، وشكراً لكم.",
]

CARS = ["تويوتا كامري", "هيونداي أكسنت", "فورد تورس", "نيسان صني", "شيفروليه ماليبو"]
DISTRICTS = ["النرجس", "الملقا", "الروضة", "السلامة", "الشفا", "الخالدية", "العزيزية"]


def _slots(rng: random.Random) -> dict[str, str]:
    return {
        "city": rng.choice(CITIES),
        "district": rng.choice(DISTRICTS),
        "car": rng.choice(CARS),
        "year": str(rng.randint(2012, 2024)),
        "date": f"{rng.randint(1, 29)} {rng.choice(HIJRI_MONTHS)} 14{rng.randint(40, 46)}هـ",
        "amount": f"{rng.randint(3, 400) * 250:,}",
        "case": f"{rng.randint(10, 99)}{rng.randint(100000, 999999)}",
        "count": str(rng.randint(3, 40)),
    }


def _wrap(rng: random.Random, org: str, subject: str, body: str) -> str:
    slots = _slots(rng)
    name = f"{rng.choice(FIRST_NAMES)} {rng.choice(FAMILY_NAMES)}"
    national_id = f"1{rng.randint(0, 9)}{rng.randint(10**7, 10**8 - 1)}"

    opening = rng.choice(OPENINGS).format(org=org, subject=subject)
    closing = rng.choice(CLOSINGS)
    body = body.format(**slots)

    lines = [opening, "", f"الموضوع: {subject}" if "الموضوع" not in opening else "", body, ""]
    if rng.random() < 0.8:
        lines += [f"مقدمه: {name}", f"رقم الهوية: {national_id}"]
    if rng.random() < 0.5:
        lines.append(f"رقم الجوال: 05{rng.randint(10**7, 10**8 - 1)}")
    lines += [f"التاريخ: {slots['date']}", "", closing]
    return "\n".join(line for line in lines if line != "")


_OCR_CONFUSIONS = {"ا": "أ", "ه": "ة", "د": "ذ", "ر": "ز", "ب": "ت", "ع": "غ", "س": "ش"}


def add_ocr_noise(text: str, rng: random.Random, rate: float = 0.02) -> str:
    """Perturb text the way an Arabic OCR pass would.

    Character confusions between visually similar letters, occasional dropped
    characters, and stray line breaks — enough to check that downstream code
    does not quietly assume clean input.
    """
    out = []
    for ch in text:
        r = rng.random()
        if r < rate and ch in _OCR_CONFUSIONS:
            out.append(_OCR_CONFUSIONS[ch])
        elif r < rate * 1.4 and ch.isalpha():
            continue
        elif r < rate * 1.6 and ch == " ":
            out.append("\n")
        else:
            out.append(ch)
    return "".join(out)


def generate_templates(
    n_per_class: int = 20,
    seed: int = 7,
    taxonomy: Taxonomy | None = None,
    hard_case_ratio: float = 0.15,
    ocr_noise_ratio: float = 0.25,
) -> list[Document]:
    """Offline generator. Deterministic for a given seed."""
    tax = taxonomy or load_taxonomy()
    rng = random.Random(seed)
    docs: list[Document] = []

    for inst in tax.institutions:
        scenarios = SCENARIOS.get(inst.id)
        if not scenarios:
            raise KeyError(f"no template scenarios defined for institution {inst.id!r}")
        for i in range(n_per_class):
            subject, body = scenarios[i % len(scenarios)]
            text = _wrap(rng, inst.name_ar, subject, body)
            scanned = rng.random() < ocr_noise_ratio
            if scanned:
                text = add_ocr_noise(text, rng)
            docs.append(
                Document(
                    doc_id=f"{inst.id}-{i:03d}",
                    text=text,
                    source="ocr" if scanned else "mock",
                    true_label=inst.id,
                )
            )

    n_hard = int(len(docs) * hard_case_ratio)
    for i in range(n_hard):
        label, subject, body = HARD_CASES[i % len(HARD_CASES)]
        org = tax.name_ar(label)
        text = _wrap(rng, org, subject, body)
        docs.append(Document(doc_id=f"hard-{i:03d}", text=text, source="mock", true_label=label))

    rng.shuffle(docs)
    return docs


GENERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "documents": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "text": {"type": "string"},
                    "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                },
                "required": ["subject", "text", "difficulty"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["documents"],
    "additionalProperties": False,
}


def generate_llm(
    n_per_class: int = 10,
    taxonomy: Taxonomy | None = None,
    model: str | None = None,
    client: anthropic.Anthropic | None = None,
    ocr_noise_ratio: float = 0.25,
    seed: int = 7,
) -> list[Document]:
    """Ask Claude for realistic correspondence, one request per institution."""
    tax = taxonomy or load_taxonomy()
    client = client or anthropic.Anthropic()
    model = model or DEFAULT_MODEL
    rng = random.Random(seed)
    docs: list[Document] = []

    for inst in tax.institutions:
        neighbours = [
            tax.name_ar(b if a == inst.id else a)
            for a, b in tax.confusion_pairs
            if inst.id in (a, b)
        ]
        overlap = (
            f"اجعل نحو ثلثها من نوع «hard»: وثائق تختص بها {inst.name_ar} فعلاً "
            f"لكنها تحمل ألفاظاً توحي بـ {'، أو '.join(neighbours)}."
            if neighbours
            else "اجعل نحو ثلثها من نوع «hard»: صياغة غير مباشرة وموضوع غير معلن في العنوان."
        )
        prompt = (
            f"اكتب {n_per_class} وثيقة حكومية سعودية واقعية موجهة إلى: {inst.name_ar}.\n"
            f"اختصاص الجهة: {inst.description_ar}\n"
            f"أمثلة على أنواع الوثائق: {'، '.join(inst.document_types_ar)}\n\n"
            "المتطلبات:\n"
            "- عربية فصحى إدارية، بأسلوب المعاملات الرسمية السعودية.\n"
            "- نوّع بين خطاب مواطن، وخطاب منشأة، ومذكرة داخلية، ومحضر.\n"
            "- بين 80 و250 كلمة لكل وثيقة، مع عناصر واقعية: تاريخ هجري، رقم هوية "
            "أو سجل، مدينة، مرفقات.\n"
            "- استخدم أسماء وأرقاماً وهمية بالكامل.\n"
            f"- {overlap}\n"
            "- لا تذكر اسم الجهة في متن الوثيقة أكثر من مرة واحدة على الأكثر، "
            "ولا تجعل التصنيف بديهياً من كلمة واحدة."
        )

        response = client.messages.create(
            model=model,
            max_tokens=16000,
            output_config={"format": {"type": "json_schema", "schema": GENERATION_SCHEMA}},
            messages=[{"role": "user", "content": prompt}],
        )
        text = next(b.text for b in response.content if b.type == "text")
        payload = json.loads(text)

        for i, item in enumerate(payload["documents"]):
            body = item["text"]
            scanned = rng.random() < ocr_noise_ratio
            if scanned:
                body = add_ocr_noise(body, rng)
            docs.append(
                Document(
                    doc_id=f"{inst.id}-llm-{i:03d}",
                    text=body,
                    source="ocr" if scanned else "mock",
                    true_label=inst.id,
                )
            )

    rng.shuffle(docs)
    return docs


CURATED_DIR = Path(__file__).resolve().parents[2] / "data" / "curated"


def generate_curated(
    taxonomy: Taxonomy | None = None,
    directory: str | Path | None = None,
    ocr_noise_ratio: float = 0.25,
    seed: int = 7,
    hard_only: bool = False,
) -> list[Document]:
    """Load the hand-authored corpus shipped in `data/curated/`.

    Written to read like real correspondence rather than drawn from
    per-institution phrase pools, so unlike the template engine it does not
    hand a keyword model the answer. Includes adversarial boundary cases —
    two per declared confusion pair — labeled with the institution that is
    actually competent while carrying the surface signals of its partner.

    Free, offline and deterministic: no API key required.
    """
    tax = taxonomy or load_taxonomy()
    directory = Path(directory) if directory else CURATED_DIR
    files = sorted(directory.glob("corpus_part*.yaml"))
    if not files:
        raise FileNotFoundError(f"no corpus_part*.yaml found in {directory}")

    rng = random.Random(seed)
    known = set(tax.ids)
    docs: list[Document] = []
    seen: set[str] = set()

    for file in files:
        payload = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
        for entry in payload.get("documents", []):
            label = entry["label"]
            if label not in known:
                raise ValueError(f"{file.name}: unknown label {label!r} on {entry['id']}")
            if entry["id"] in seen:
                raise ValueError(f"duplicate document id {entry['id']!r}")
            seen.add(entry["id"])

            if hard_only and entry.get("difficulty") != "hard":
                continue

            text = entry["text"].strip()
            scanned = rng.random() < ocr_noise_ratio
            if scanned:
                text = add_ocr_noise(text, rng)
            docs.append(
                Document(
                    doc_id=entry["id"],
                    text=text,
                    source="ocr" if scanned else "mock",
                    true_label=label,
                )
            )

    missing = known - {d.true_label for d in docs}
    if missing and not hard_only:
        raise ValueError(f"curated corpus has no documents for: {sorted(missing)}")

    rng.shuffle(docs)
    return docs


def save_jsonl(docs: list[Document], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(doc.model_dump_json() + "\n")
    return path


def load_jsonl(path: str | Path) -> list[Document]:
    with Path(path).open(encoding="utf-8") as fh:
        return [Document.model_validate_json(line) for line in fh if line.strip()]


def save_as_files(docs: list[Document], directory: str | Path) -> Path:
    """Write documents as .txt files, one per document, foldered by label.

    Useful for exercising the ingestion path end to end rather than feeding
    the classifier from JSONL.
    """
    directory = Path(directory)
    for doc in docs:
        folder = directory / (doc.true_label or "unlabeled")
        folder.mkdir(parents=True, exist_ok=True)
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", doc.doc_id)
        (folder / f"{safe}.txt").write_text(doc.text, encoding="utf-8")
    return directory
