# Improved RAG Evaluation Framework Plan

## Current Issues with Test Cases

### 1. **URL Format Normalization**
- **Problem**: `linkedin.com/company/hubspot` vs `linkedin.com/hubspot` both valid
- **Solution**: Normalize URLs before comparison, accept both formats

### 2. **Contact Information Flexibility**
- **Problem**: Finding additional valid emails gets penalized
- **Solution**: 
  - Award partial credit for finding any valid company email
  - Use fuzzy matching for similar contact info
  - Focus on finding *any* legitimate contact rather than exact matches

### 3. **Key People Extraction**
- **Problem**: Extracting generic mentions instead of actual names
- **Solution**: 
  - Improve NER (Named Entity Recognition) for person names
  - Focus on executive titles + names in same context
  - Use more sophisticated pattern matching

### 4. **Phone Number Normalization**
- **Problem**: Different formatting of same number
- **Solution**: Strip formatting and compare numeric values

### 5. **More Realistic Expectations**
- **Problem**: Some expected info may not be publicly available
- **Solution**: 
  - Manually verify all expected results are actually findable
  - Create tiered scoring (required vs nice-to-have)
  - Add difficulty-based scoring weights

## Proposed Improvements

### 1. **Flexible Matching Algorithm**
```python
def flexible_email_match(expected, actual):
    # Match domain if exact email not found
    expected_domain = expected.split('@')[1]
    actual_domain = actual.split('@')[1]
    if expected_domain == actual_domain:
        return 0.7  # Partial credit for same domain
    return 0.0

def normalize_linkedin_url(url):
    # Remove /company/ variations, normalize formats
    return url.replace('/company/', '/').replace('https://', '').replace('www.', '')
```

### 2. **Tiered Scoring System**
- **Essential** (1.0 weight): Company website, primary social media
- **Important** (0.8 weight): Support email, main phone number
- **Nice-to-have** (0.5 weight): Press contacts, specific executives

### 3. **Verification of Test Cases**
- Manually verify each expected result is actually findable
- Update outdated information
- Remove unrealistic expectations

### 4. **Better Key People Extraction**
- Look for patterns like "CEO John Smith" rather than "CEO mentioned in context"
- Use spaCy or similar for proper name extraction
- Focus on executive team pages, about pages

### 5. **Domain-Specific Evaluation**
- Different scoring for different industries
- B2B companies vs B2C companies have different contact visibility
- Adjust expectations based on company size and public presence

## Implementation Priority

1. **High Priority**: Fix key people extraction and URL normalization
2. **Medium Priority**: Implement flexible matching and tiered scoring
3. **Low Priority**: Manually verify and update all test cases

## Success Metrics

- Target: 60%+ success rate on realistic test cases
- Current: 0% success rate (too strict evaluation)
- Improved: Should see 40-70% success rate with fairer evaluation 