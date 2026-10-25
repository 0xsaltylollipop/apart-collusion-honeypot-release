# Attribution

## Demand and benchmark code

`collusion/market/economics.py` adapts the logit demand, profit and Nash/monopoly benchmark logic from
**algorithmic_pricing_llms** (https://github.com/luciasauer/algorithmic_pricing_llms,
`src/environment/pricing_market_logic_multiproduct.py`), simplified to plain (non-nested) logit demand.
That code is released under the MIT licence:

> MIT License
>
> Copyright (c) 2025 Moritz Peist, Julian Romero & Lucia Sauer
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

## Market design and prompts (cited; prompt text and one parsing rule reproduced)

- Demand model and parameters: Calvano, Calzolari, Denicolò & Pastorello (2020), *Artificial Intelligence,
  Algorithmic Pricing, and Collusion*, AER; Fish, Gonczarowski & Shorrer (2024), arXiv 2404.00806.
- P0 prompt, initial prior round, history window, price ceiling and the 3-firm parameters: Riemer et al.
  (2026), *Position: Collusion Risks Among AI Reasoning Agents Justify Certification Requirements for Making
  Market Decisions*, ICML 2026 (LLMCartel). Their repository has no licence. For comparability we reproduce their
  P0 prompt text verbatim, and `parse_price_lcartel_gpt` (collusion/game/prompts.py) reproduces their GPT price
  rule (`extract_price` in `chatgpt_duopoly_tester.py`: three regular expressions, first match wins);
  `parse_price_lcartel` re-implements their R1 parsing rule. No other code from it is used. See `docs/prompts.md`.
  The reproduced P0 prompt text (`P0_SYSTEM` and the P0 wording in `collusion/game/prompts.py`) and the
  reproduced GPT price rule remain the work of Riemer et al.; this repository's MIT licence (`LICENSE`) does not
  cover them.
