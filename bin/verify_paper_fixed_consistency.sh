#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

verify_one() {
  local source_json="$1"
  local script_path="$2"
  local n_value="$3"
  local tmp_json="/tmp/verify_${source_json}"

  echo "Verifying ${source_json}"
  rye run python "$script_path" --n "$n_value" --only-fixed --output "$tmp_json"

  perl -MJSON::PP -e '
sub load_json {
  my ($path) = @_;
  open my $fh, "<", $path or die $!;
  local $/;
  return decode_json(<$fh>);
}

sub cmp_num {
  my ($a, $b) = @_;
  return abs($a - $b) < 1e-12;
}

sub cmp_any {
  my ($a, $b) = @_;
  my $ra = ref $a;
  my $rb = ref $b;
  return 0 if $ra ne $rb;

  if (!$ra) {
    return ($a =~ /^-?[0-9.]+(?:e[+-]?[0-9]+)?$/i && $b =~ /^-?[0-9.]+(?:e[+-]?[0-9]+)?$/i)
      ? cmp_num($a, $b)
      : $a eq $b;
  }

  if ($ra eq "ARRAY") {
    return 0 unless @$a == @$b;
    for my $i (0 .. $#$a) {
      return 0 unless cmp_any($a->[$i], $b->[$i]);
    }
    return 1;
  }

  if ($ra eq "HASH") {
    my @ka = sort keys %$a;
    my @kb = sort keys %$b;
    return 0 unless "@ka" eq "@kb";
    for my $k (@ka) {
      return 0 unless cmp_any($a->{$k}, $b->{$k});
    }
    return 1;
  }

  return 0;
}

my $orig = load_json($ARGV[0]);
my $tmp = load_json($ARGV[1]);

for my $key (sort keys %$orig) {
  die "missing key: $key\n" unless exists $tmp->{$key};
  for my $method (qw(so po)) {
    die "mismatch: $key/$method\n" unless cmp_any($orig->{$key}{$method}, $tmp->{$key}{$method});
  }
}

print "OK: so/po match for $ARGV[0]\n";
' "$source_json" "$tmp_json"
}

verify_one "paper_10_300.json" "src/main_exp_test_par copy.py" 300
verify_one "paper_10_1000.json" "src/main_exp_test_par copy.py" 1000
verify_one "paper_300.json" "src/main_exp_test_par.py" 300
verify_one "paper_1000.json" "src/main_exp_test_par.py" 1000
