<?php

//echo 'here';
if($_SERVER['REQUEST_METHOD']=='POST'){
$image = $_POST['image'];
//$image = base64_decode($image);
//echo $image;
//return;
$pwd  = '/home/jarvis/normal/jarvisBoardCode/uploads/';
$path = $pwd."img3.jpg";
//echo $path;
//file_put_contents($path,$image);
file_put_contents($path,base64_decode($image));

//echo $_FILES['image']['tmp_name'];
//    if(move_uploaded_file($_FILES['image']['tmp_name'], $path) ){
//        echo "success";
//    } else{
//        echo "fail";
//    }
//echo "Successfully Uploaded";
$pwd  = '/home/jarvis/normal/jarvisBoardCode/';
$cmd ='python '. $pwd.'inference.py';
$cmdArgs = ' uploads/img3.jpg';
$cmd .= $cmdArgs;

//$outp = exec('rm '.$pwd.'/predictions.txt', $outp, $return);
//$outp = exec('chmod u+s '.$pwd.'/predictions.txt', $outp, $return);
//echo $cmd;
$output = exec($cmd, $output, $return);
if (!$return) {
//    echo $cmd;
//    echo "yay!";
} else{
//    echo $output;
    $ankit = exec('cat '.$pwd.'predictions.txt', $ankit, $return);
//    echo $return;
    echo $ankit;
    return;
}
}
else {
$pwd  = '/home/jarvis/normal/jarvisBoardCode/';
$cmd ='python '. $pwd.'inference.py';
$cmdArgs = ' img2.jpg';
$cmd .= $cmdArgs;

$output = exec($cmd, $output, $return);
if (!$return) {
    echo $cmd;
    echo "yay!";
} else {
    $output = exec('cat '.$pwd.'predictions.txt', $output, $return);
    echo $output;
}
}
?>
