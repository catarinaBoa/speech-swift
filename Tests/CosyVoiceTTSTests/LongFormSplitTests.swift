import XCTest
@testable import CosyVoiceTTS

/// Unit tests for `CosyVoiceTTSModel.splitForLongForm`, the helper that breaks
/// long input text into LLM-friendly segments. No model load needed — this is
/// a pure string-processing function.
final class LongFormSplitTests: XCTestCase {

    private func split(_ text: String) -> [String] {
        CosyVoiceTTSModel.splitForLongForm(text)
    }

    // MARK: - Trivial cases

    func testSingleSentenceReturnsItself() {
        let segments = split("This is a single short sentence.")
        XCTAssertEqual(segments, ["This is a single short sentence."])
    }

    func testEmptyInputReturnsEmpty() {
        XCTAssertEqual(split("").count, 0)
        XCTAssertEqual(split("   \n\t ").count, 0)
    }

    func testNoPunctuationFallsBackToSingleSegment() {
        let text = "this has no terminating punctuation"
        XCTAssertEqual(split(text), [text])
    }

    // MARK: - Sentence-terminator splits
    //
    // Each fixture sentence is ≥ 4 words so the short-fragment merge logic
    // doesn't collapse them into a single segment.

    func testSplitsOnPeriod() {
        let s = split("This is the first sentence here. " +
                      "This is the second sentence here. " +
                      "This is the third sentence here.")
        XCTAssertEqual(s.count, 3)
        XCTAssertEqual(s.first, "This is the first sentence here.")
        XCTAssertEqual(s.last,  "This is the third sentence here.")
    }

    func testSplitsOnQuestionMark() {
        let s = split("Is this really truly a question? " +
                      "Yes it really is a statement. " +
                      "And another small follow-up question?")
        XCTAssertEqual(s.count, 3)
        XCTAssertTrue(s.first!.hasSuffix("?"))
        XCTAssertTrue(s.last!.hasSuffix("?"))
    }

    func testSplitsOnExclamation() {
        let s = split("Hello there how nice to see! " +
                      "How are you doing today? " +
                      "I am doing fine thanks for asking.")
        XCTAssertEqual(s.count, 3)
        XCTAssertTrue(s.first!.hasSuffix("!"))
    }

    func testTrailingTextWithoutTerminatorIsKept() {
        // The buffer after the last terminator must not be dropped.
        // Both fragments long enough to avoid the merge logic.
        let s = split("This is the first complete sentence here. " +
                      "Trailing fragment without any period at the end")
        XCTAssertEqual(s.count, 2)
        XCTAssertEqual(s.first, "This is the first complete sentence here.")
        XCTAssertEqual(s.last, "Trailing fragment without any period at the end")
    }

    // MARK: - Short-segment merging

    func testShortLeadFragmentMergesIntoNext() {
        // "Hi." is 1 word (< minWordsPerSegment=4) and should merge forward.
        let s = split("Hi. This is the second sentence which is longer.")
        XCTAssertEqual(s.count, 1)
        XCTAssertEqual(s[0], "Hi. This is the second sentence which is longer.")
    }

    func testTwoVeryShortSegmentsMerge() {
        let s = split("Ok. Sure. Now this one is the third sentence with enough words.")
        // "Ok." (1 word) merges forward into "Sure." (still short, merges into the next).
        XCTAssertLessThan(s.count, 3)
        XCTAssertTrue(s.last!.contains("third sentence"))
    }

    // MARK: - Long-segment clause splits

    func testLongSentenceSplitsOnComma() {
        // 30+ words exceeds maxWordsPerSegment=25, so the splitter should look for
        // clause boundaries (commas).
        let long = "This is a very long sentence with many many many many many words separated by clauses, " +
                   "and we want it broken up at sensible boundaries, before the model loses coherence."
        let s = split(long)
        XCTAssertGreaterThan(s.count, 1, "Long sentence should split into multiple clauses")
        for seg in s {
            XCTAssertLessThanOrEqual(
                seg.split(whereSeparator: { $0.isWhitespace }).count, 25,
                "Each clause should stay below the 25-word cap: \(seg)")
        }
    }

    // MARK: - Multi-paragraph realistic input

    func testRealisticLongFormInput() {
        let text = """
        Hi, this is an extended demonstration of zero-shot voice cloning running entirely on Apple Silicon. \
        Everything you are hearing was synthesized in real time. There was no fine-tuning, no cloud calls. \
        We think the next year of audio software is going to look very different.
        """
        let s = split(text)
        XCTAssertGreaterThanOrEqual(s.count, 3, "Should produce at least 3 sentence-level segments")
        // The concatenation should preserve all the text (minus whitespace normalisation).
        let joined = s.joined(separator: " ")
        XCTAssertTrue(joined.contains("Apple Silicon"))
        XCTAssertTrue(joined.contains("very different"))
    }

    // MARK: - Idempotency

    func testRepeatedSplitsAreStable() {
        let text = "One sentence. Two sentences. Three sentences."
        let first = split(text)
        // Joining and re-splitting should produce the same partitions.
        let joined = first.joined(separator: " ")
        let second = split(joined)
        XCTAssertEqual(first, second)
    }
}
