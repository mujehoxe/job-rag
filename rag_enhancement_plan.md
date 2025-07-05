# RAG System Enhancement Plan

## Current Performance Analysis

### Issues Identified:
1. **Success Rate: 0.00%** - No test cases met the 0.5 F1 threshold
2. **AI Analysis Failures** - g4f API timeouts preventing strategic search adaptation
3. **Contact Extraction Problems**:
   - Phone F1: 0.000 (extracting noise like dates, IDs)
   - LinkedIn F1: 0.000 (not finding correct profiles)
   - Email F1: 0.100 (low accuracy)
   - Social F1: 0.640 (working well)
4. **Search Strategy Issues** - Fallback to generic "Find contact" queries

### Strengths:
- High completeness score (0.91) - finding many documents
- Social media extraction working reasonably well
- Search engines working properly (Brave, Serper, Google)

## Enhancement Strategies

### 1. Fix AI Analysis System (Critical)

**Problem:** g4f API timeouts breaking strategic search
**Solutions:**
- Reduce AI analysis timeout from 20s to 10s
- Add retry logic with exponential backoff
- Implement fallback to simpler analysis without AI
- Use local LLM or switch to OpenAI API for analysis

### 2. Improve Contact Information Extraction

**Phone Number Extraction:**
- Tighten regex pattern to exclude dates/IDs
- Add country code validation
- Filter out numbers that are clearly not phone numbers
- Use context clues (words like "phone", "call", "tel")

**Email Address Extraction:**
- Improve domain filtering (exclude common non-business domains)
- Add context-based scoring (emails near "contact" text)
- Filter out automated/system emails

**LinkedIn Profile Extraction:**
- Target specific LinkedIn URLs in search queries
- Improve LinkedIn URL extraction patterns
- Distinguish between company and personal profiles

### 3. Optimize Search Query Generation

**Current Issue:** Generic fallback queries
**Improvements:**
- Add company domain-specific searches ("site:company.com contact")
- Use industry-specific search terms
- Add location-based searches for contact info
- Target specific platforms (LinkedIn, company websites)

### 4. Add Domain-Specific Targeting

**Company Website Prioritization:**
- Always search company's main domain first
- Look for /about, /contact, /team pages specifically
- Extract structured data from company pages

**Social Media Targeting:**
- Direct searches on LinkedIn, Twitter, Facebook
- Use platform-specific search operators
- Extract verified account information

### 5. Improve Document Processing

**Content Quality Filtering:**
- Prioritize official company pages over news articles
- Filter out irrelevant content (ads, general articles)
- Focus on business-relevant sections

**Extraction Context:**
- Use surrounding text context for better extraction
- Implement named entity recognition
- Add confidence scoring for extracted information

## Implementation Priority

### Phase 1: Critical Fixes (Immediate)
1. **Fix AI Analysis Timeouts**
   - Reduce timeout to 10s
   - Add retry logic
   - Implement fallback analysis

2. **Improve Phone Number Extraction**
   - Better regex patterns
   - Context-based filtering
   - Validation logic

### Phase 2: Search Enhancement (Short-term)
1. **Domain-Specific Searches**
   - Add site: operators
   - Target company domains
   - Use structured search patterns

2. **LinkedIn Targeting**
   - Specific LinkedIn searches
   - Better profile extraction
   - Company vs personal profile distinction

### Phase 3: Advanced Features (Medium-term)
1. **Content Quality Scoring**
   - Relevance scoring
   - Source reliability
   - Information confidence

2. **Multi-stage Extraction**
   - Initial broad search
   - Targeted refinement
   - Cross-validation

## Expected Improvements

### Target Metrics:
- **Success Rate:** 0% → 60%+ (by fixing AI analysis and extraction)
- **Email F1:** 0.1 → 0.7+ (better filtering and context)
- **Phone F1:** 0.0 → 0.5+ (proper regex and validation)
- **LinkedIn F1:** 0.0 → 0.6+ (targeted searches)
- **Overall F1:** 0.18 → 0.6+ (combined improvements)

### Performance Goals:
- Reduce search time by 30% (better AI analysis)
- Increase relevant documents by 50% (targeted searches)
- Improve extraction accuracy by 70% (better patterns)

## Testing Strategy

1. **Unit Tests** - Test individual extraction components
2. **Integration Tests** - Test search and extraction pipeline
3. **Regression Tests** - Ensure improvements don't break existing functionality
4. **Evaluation Framework** - Regular runs with full test suite

## Implementation Timeline

- **Week 1:** Fix AI analysis and phone extraction
- **Week 2:** Improve search targeting and LinkedIn extraction
- **Week 3:** Add domain-specific searches and content filtering
- **Week 4:** Testing, optimization, and final validation

## Success Metrics

- Overall F1 score > 0.6
- Success rate > 60%
- All contact types (email, phone, social, LinkedIn) > 0.5 F1
- Average execution time < 120 seconds
- Documents found relevance > 80% 