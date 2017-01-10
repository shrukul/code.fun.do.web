<?php
$output = exec('ls /', $output, $return);

if (!$return) {
    echo $output;
    echo "yay!";
} else {
    echo "PDF Created Successfully";
}
?>
