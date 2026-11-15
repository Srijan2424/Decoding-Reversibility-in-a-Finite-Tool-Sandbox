import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.Base64;
import java.util.HexFormat;
import java.util.TreeSet;
import scienceworld.runtime.pythonapi.PythonInterface;

/** Runs a bounded CPU diagnostic in the pinned simulator JVM. */
public final class RecoveryReplayBench {
    private final PythonInterface simulator = new PythonInterface();
    private final String reference;
    private final String candidate;
    private final String[] firstCommands;
    private final MessageDigest sha = MessageDigest.getInstance("SHA-256");

    private RecoveryReplayBench() throws Exception {
        simulator.load("freeze", 4, "", false);
        reset();
        reference = simulator.getObjectTree("");
        candidate = Arrays.stream(commands()).filter(a -> a.startsWith("open ") && a.contains("door")).findFirst().orElseThrow();
        reset();
        simulator.step(candidate);
        firstCommands = commands();
    }

    private void reset() {
        simulator.reset();
        simulator.step("look around");
    }

    private boolean done() {
        return simulator.getCompleted() || simulator.getNumMoves() > 250 ||
                Math.round(simulator.getScore() * 100) < 0;
    }

    private String[] commands() {
        return new TreeSet<>(simulator.getValidActionObjectCombinations()).toArray(String[]::new);
    }

    private String stateHash() {
        return HexFormat.of().formatHex(sha.digest(simulator.getObjectTree("").getBytes(StandardCharsets.UTF_8)));
    }

    private void run(boolean benchmark, int secondLimit, long seconds) {
        long start = System.nanoTime();
        long deadline = start + seconds * 1_000_000_000L;
        int checkedFirst = 0, checkedSecond = 0;
        long listedSecond = 0;
        boolean interruptedFirst = false;
        for (String first : firstCommands) {
            if (System.nanoTime() >= deadline) {
                interruptedFirst = true;
                break;
            }
            reset();
            simulator.step(candidate);
            simulator.step(first);
            checkedFirst++;
            if (benchmark) {
                System.out.printf("CHECK\t%s\t%s\t%s\t%s%n",
                        Base64.getEncoder().encodeToString(first.getBytes(StandardCharsets.UTF_8)),
                        stateHash(), done(), reference.equals(simulator.getObjectTree("")));
            } else if (reference.equals(simulator.getObjectTree(""))) {
                System.out.println("MATCH\t" + first);
            }
            String[] secondCommands = done() ? new String[0] : commands();
            listedSecond += secondCommands.length;
            if (benchmark) continue;
            for (String second : secondCommands) {
                if (checkedSecond >= secondLimit || System.nanoTime() >= deadline) break;
                reset();
                simulator.step(candidate);
                simulator.step(first);
                if (done()) throw new IllegalStateException("Terminal status changed on replay");
                simulator.step(second);
                checkedSecond++;
                if (reference.equals(simulator.getObjectTree(""))) {
                    System.out.println("MATCH2\t" + first + "\t" + second);
                }
            }
        }
        System.out.printf("SUMMARY\t%d\t%d\t%d\t%d\t%.3f\t%b%n",
                firstCommands.length, checkedFirst, listedSecond, checkedSecond,
                (System.nanoTime() - start) / 1e9, interruptedFirst);
    }

    public static void main(String[] args) throws Exception {
        new RecoveryReplayBench().run(args[0].equals("benchmark"),
                Integer.parseInt(args[1]), Long.parseLong(args[2]));
    }
}
