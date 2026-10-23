# F2 label guide: document type

**Task.** Given the text of **one page** (OCR'd or extracted from a PDF), output exactly one
label. The model outputs only the label, nothing else.

**Unit = one page.** A PDF often bundles several documents (bill + discharge summary +
settlement letter). We classify page by page; a continuation page (bill page 3, the second
page of a letter) gets the label of the document it belongs to. *(My call: page-level keeps
the input short for a 0.5B model and matches how FairClaims sorts uploads.)*

## Labels

### `settlement_letter`
- **Definition:** the insurer or TPA telling the policyholder or hospital that a claim has
  been **paid, in full or in part**.
- **Tells:** "claim settled", "amount approved / paid", "UTR / NEFT reference", a table of
  claimed vs approved vs deducted amounts.
- **Confused with:** `rejection_letter`. **Rule: if any amount was paid, it's a settlement
  letter**, even when most of the claim was deducted. *(My call: a partial settlement is a
  settlement; underpayment is FairClaims' "I was paid less" route.)*

### `rejection_letter`
- **Definition:** the insurer or TPA refusing a **reimbursement or final claim** outright,
  with ₹0 paid.
- **Tells:** "repudiated", "claim is not payable", "regret to inform", a clause or exclusion
  cited as the reason.
- **Confused with:** `preauth_denial` (that's about cashless *before or during*
  treatment) and `query_letter` (that asks for more information; nothing is decided yet).

### `preauth_denial`
- **Definition:** the insurer or TPA denying or limiting **cashless authorisation** for a
  planned or ongoing hospitalisation.
- **Tells:** "pre-authorisation", "cashless request", "not approved for cashless",
  "patient may claim reimbursement", usually addressed to the hospital.
- **Confused with:** `rejection_letter`. A cashless denial is **not** a claim rejection; the
  patient can still file for reimbursement. *(This distinction matters for routing: it's
  FairClaims' "Cashless was denied" path.)*

### `query_letter` *(added)*
- **Definition:** the insurer or TPA **asking for documents or clarification** before
  deciding the claim (often called a deficiency letter).
- **Tells:** "please submit", "documents required", "deficiency", "within N days",
  "claim will be closed if not received".
- **Confused with:** `rejection_letter`. A "closure for non-submission" letter is a
  rejection letter, because a decision was made. *(Added because claims commonly stall at
  this stage, and the right next step, sending the documents, is different from appealing.)*

### `policy_schedule`
- **Definition:** the document stating **what this specific policy covers**: insured
  members, sum insured, dates, plan, add-ons. Includes the Customer Information Sheet.
- **Tells:** policy number, period of insurance, sum insured, list of insured members,
  premium, room-rent limit, co-pay, waiting periods.
- **Confused with:** the generic policy wording (the full terms and conditions booklet). *(My
  call: policy wording pages are `other` for now. They're long and generic, and they need
  their own task, P3.)*

### `hospital_bill`
- **Definition:** any **bill, invoice or receipt for treatment costs**: the final hospital
  bill, an itemised breakdown, pharmacy invoices, lab or diagnostic invoices.
- **Tells:** line items with amounts, totals, GST, "final bill", "IP bill", "invoice no.".
- **Confused with:** `medical_report`. A lab *invoice* is a bill; the lab *results* are a
  report. *(My call: one bill type, not separate pharmacy and lab types. F3 classifies the
  lines inside bills later.)*

### `discharge_summary`
- **Definition:** the hospital's clinical summary of a stay, written at discharge.
- **Tells:** admission and discharge dates, diagnosis, history of present illness, procedure
  done, treatment given, condition at discharge, advice on discharge.
- **Confused with:** `medical_report`. A discharge summary covers the **whole stay**; a
  report covers one test or one visit.

### `medical_report` *(added)*
- **Definition:** a clinical document that is **not** a discharge summary: lab or radiology
  results, prescriptions, consultation notes, investigation reports.
- **Tells:** test names with values and reference ranges, "impression", "Rx", a doctor's
  consultation notes.
- **Confused with:** `discharge_summary` (see above) and `hospital_bill` (results vs prices).
  *(Added because these pages are common in claim uploads and often decide pre-existing
  disease arguments, so they shouldn't vanish into `other`.)*

### `health_card`
- **Definition:** an insurance or TPA ID card, e-card, or a government health ID
  (e.g. ABHA, Ayushman card).
- **Tells:** very short text: name, member or card ID, policy number, TPA helpline, validity.
- **Confused with:** `policy_schedule`. A card has an ID but no coverage details or premium.

### `other`
- **Definition:** anything relevant to a claim that doesn't fit above: policy wording
  booklets, KYC or ID proof (Aadhaar, PAN), cancelled cheques, claim forms, emails or chats,
  blank or unreadable pages.
- **Rule:** use `other` only when you can name which item on this list the page is. If
  you're unsure between two real labels, pick the closer one; **`other` is not for hard cases.**

## Edge cases

| Case | Label | Why |
|---|---|---|
| Settlement letter paying ₹1,200 of a ₹1.5L claim | `settlement_letter` | Something was paid |
| Letter: "claim closed as documents not received" | `rejection_letter` | A decision was made |
| Cashless denied, then reimbursement rejected later | two documents: `preauth_denial`, `rejection_letter` | Different stages |
| Bill summary page with no line items | `hospital_bill` | Still a bill |
| Policy schedule with a printed ID card at the bottom | `policy_schedule` | Pick the main content of the page |
| Blank page, or OCR garbage | `other` | Nothing to classify |
| Email forwarding a settlement letter, letter text included | `settlement_letter` | Classify the content, not the wrapper |

## Data notes for Step 2

- Mix the **sources**: insurer letters, TPA letters (Medi Assist, Paramount, etc. as
  *styles*, not copies), hospital documents from big chains and small nursing homes.
- Mix the **quality**: clean digital PDFs, OCR noise (broken words, misread digits), partial
  pages, Hindi or regional text mixed into English documents.
- **Balance the labels** in training. Real uploads are mostly bills and summaries, so also
  check the model on a test set skewed like real uploads.
