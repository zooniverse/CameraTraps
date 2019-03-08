#!/bin/bash

EXPERIMENT_DIR=/Users/sarabeery/Documents/CameraTrapClass/sim_classification/general/train_on_cct

DATABASE_DIR=/Users/sarabeery/Documents/CameraTrapClass/Fixing_CCT_Anns/Corrected_versions

SPLIT_FILE=eccv_18_train_test_split_with_imerit.json

DB_FILE=CombinedBBoxAndECCV18.json

RESULTS=eval_all_results.npz

EXP=cis_test

python evaluate_cropped_box_classification.py --database_json_file $DATABASE_DIR/$DB_FILE --results_file $EXPERIMENT_DIR/results/$RESULTS --alternate_category_file $DATABASE_DIR/"eccv_categories.json" --save_folder $EXPERIMENT_DIR/results/ --data_split_file $DATABASE_DIR/$SPLIT_FILE --data_split_to_evaluate $EXP

EXP=cis_val

python evaluate_cropped_box_classification.py --database_json_file $DATABASE_DIR/$DB_FILE --results_file $EXPERIMENT_DIR/results/$RESULTS --alternate_category_file $DATABASE_DIR/"eccv_categories.json" --save_folder $EXPERIMENT_DIR/results/ --data_split_file $DATABASE_DIR/$SPLIT_FILE --data_split_to_evaluate $EXP

EXP=train

python evaluate_cropped_box_classification.py --database_json_file $DATABASE_DIR/$DB_FILE --results_file $EXPERIMENT_DIR/results/$RESULTS --alternate_category_file $DATABASE_DIR/"eccv_categories.json" --save_folder $EXPERIMENT_DIR/results/ --data_split_file $DATABASE_DIR/$SPLIT_FILE --data_split_to_evaluate $EXP

EXP=trans_test_with_imerit

python evaluate_cropped_box_classification.py --database_json_file $DATABASE_DIR/$DB_FILE --results_file $EXPERIMENT_DIR/results/$RESULTS --alternate_category_file $DATABASE_DIR/"eccv_categories.json" --save_folder $EXPERIMENT_DIR/results/ --data_split_file $DATABASE_DIR/$SPLIT_FILE --data_split_to_evaluate $EXP

EXP=trans_val

python evaluate_cropped_box_classification.py --database_json_file $DATABASE_DIR/$DB_FILE --results_file $EXPERIMENT_DIR/results/$RESULTS --alternate_category_file $DATABASE_DIR/"eccv_categories.json" --save_folder $EXPERIMENT_DIR/results/ --data_split_file $DATABASE_DIR/$SPLIT_FILE --data_split_to_evaluate $EXP

EXP=trans_test

python evaluate_cropped_box_classification.py --database_json_file $DATABASE_DIR/$DB_FILE --results_file $EXPERIMENT_DIR/results/$RESULTS --alternate_category_file $DATABASE_DIR/"eccv_categories.json" --save_folder $EXPERIMENT_DIR/results/ --data_split_file $DATABASE_DIR/$SPLIT_FILE --data_split_to_evaluate $EXP

EXP=imerit

python evaluate_cropped_box_classification.py --database_json_file $DATABASE_DIR/$DB_FILE --results_file $EXPERIMENT_DIR/results/$RESULTS --alternate_category_file $DATABASE_DIR/"eccv_categories.json" --save_folder $EXPERIMENT_DIR/results/ --data_split_file $DATABASE_DIR/$SPLIT_FILE --data_split_to_evaluate $EXP


